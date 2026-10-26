# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
from typing import Any

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.optim import Optimizer

from lerobot.policies.gaussian_actor.modeling_gaussian_actor import (
    DISCRETE_DIMENSION_INDEX,
    MLP,
)
from lerobot.policies.td3_actor.modeling_td3_actor import TD3ActorPolicy, TD3ResidualActor
from lerobot.types import BatchType
from lerobot.utils.constants import ACTION
from lerobot.utils.transition import move_state_dict_to_device

from ..base import RLAlgorithm
from ..configs import TrainingStats
from ..sac.sac_algorithm import CriticEnsemble, CriticHead
from .configuration_td3 import TD3_POLICY_GRADIENT_TYPES, TD3AlgorithmConfig


class TD3Algorithm(RLAlgorithm):
    """(Residual) TD3 with a REDQ critic ensemble.

    Owns the critic ensemble, the target critics and the target actor, plus the
    TD3 update loop (delayed policy updates, target-policy smoothing, optional
    Q-target clamping for sparse rewards). The deterministic residual actor and
    the observation encoder are owned by :class:`TD3ActorPolicy`.

    Batch actions are **combined** actions (base + residual) as stored in the
    replay buffer; the residual for the actor loss is reconstructed internally
    as ``combined - base_action(obs)`` via the actor module.
    """

    config_class = TD3AlgorithmConfig
    name = "td3"

    def __init__(
        self,
        policy: TD3ActorPolicy,
        config: TD3AlgorithmConfig,
    ):
        if not isinstance(policy, TD3ActorPolicy):
            raise TypeError(
                f"TD3Algorithm expects a TD3ActorPolicy, got {type(policy).__name__}. "
                "Use --policy.type=td3_actor with --algorithm.type=td3."
            )
        if config.policy_gradient_type not in TD3_POLICY_GRADIENT_TYPES:
            raise ValueError(
                f"policy_gradient_type must be one of {TD3_POLICY_GRADIENT_TYPES}, "
                f"got '{config.policy_gradient_type}'"
            )

        self.config = config
        self.policy_config = config.policy_config
        self.policy = policy
        self.optimizers: dict[str, Optimizer] = {}
        self._optimization_step: int = 0

        action_dim = self.policy.config.output_features[ACTION].shape[0]
        self.continuous_action_dim = (
            action_dim - 1 if self.policy.config.num_discrete_actions is not None else action_dim
        )
        self._init_critics()
        self._init_actor_target()

        self._device = torch.device(self.policy.config.device)
        self._move_to_device()

    def _init_critics(self) -> None:
        """Build critic ensemble and target critics."""
        encoder = self.policy.encoder_critic

        def make_ensemble() -> CriticEnsemble:
            heads = [
                CriticHead(
                    input_dim=encoder.output_dim + self.continuous_action_dim,
                    **asdict(self.config.critic_network_kwargs),
                )
                for _ in range(self.config.num_critics)
            ]
            return CriticEnsemble(encoder=encoder, ensemble=heads)

        self.critic_ensemble = make_ensemble()
        self.critic_target = make_ensemble()
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        # NOTE: torch.compile stays disabled, mirroring SAC (no convergence with compile).
        if self.config.use_torch_compile:
            self.critic_ensemble = torch.compile(self.critic_ensemble)
            self.critic_target = torch.compile(self.critic_target)

    def _init_actor_target(self) -> None:
        """Build the target actor.

        Like the ResFiT agent, the target actor head operates on features from
        the *online* (shared) encoder: only the head weights are EMA-tracked,
        so the target actor reuses the policy's encoder object instead of
        holding a frozen copy.
        """
        trunk_input_dim = self.policy.encoder_actor.output_dim
        if self.policy.config.residual:
            trunk_input_dim += self.continuous_action_dim
        self.actor_target = TD3ResidualActor(
            encoder=self.policy.encoder_actor,
            network=MLP(
                input_dim=trunk_input_dim,
                **asdict(self.policy.config.actor_network_kwargs),
            ),
            continuous_action_dim=self.continuous_action_dim,
            num_discrete_actions=self.policy.config.num_discrete_actions,
            residual=self.policy.config.residual,
            action_scale=self.policy.config.action_scale,
            init_final=self.policy.config.actor_last_layer_init_scale,
            encoder_is_shared=self.policy.shared_encoder,
        )
        self.actor_target.load_state_dict(self.policy.actor.state_dict())

    def _move_to_device(self) -> None:
        self.policy.to(self._device)
        self.critic_ensemble.to(self._device)
        self.critic_target.to(self._device)
        self.actor_target.to(self._device)

    def update(self, batch_iterator: Iterator[BatchType]) -> TrainingStats:
        """Run one TD3 training step (critic ensemble, then delayed actor).

        Pulls ``utd_ratio`` batches from ``batch_iterator``; the actor is only
        updated after ``critic_warmup_steps`` optimization steps and every
        ``policy_update_freq``-th step thereafter.

        Args:
            batch_iterator: yields batches each containing
                - ``action``: Combined action tensor (base + residual)
                - ``reward``: Reward tensor
                - ``state``: Observations tensor dict (incl. ``observation.base_action``)
                - ``next_state``: Next observations tensor dict (incl. ``observation.base_action``)
                - ``done``: Done mask tensor
                - ``observation_feature``: Optional pre-computed observation features
                - ``next_observation_feature``: Optional pre-computed next observation features

        Returns:
            TrainingStats with per-component losses and grad norms.
        """
        clip = self.config.grad_clip_norm

        for _ in range(self.config.utd_ratio - 1):
            batch = next(batch_iterator)
            fb = self._prepare_forward_batch(batch)

            loss_critic = self._compute_loss_critic(fb)
            self.optimizers["critic"].zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_ensemble.parameters(), max_norm=clip)
            self.optimizers["critic"].step()

            self._update_target_networks()

        batch = next(batch_iterator)
        fb = self._prepare_forward_batch(batch)

        loss_critic = self._compute_loss_critic(fb)
        self.optimizers["critic"].zero_grad()
        loss_critic.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(self.critic_ensemble.parameters(), max_norm=clip).item()
        self.optimizers["critic"].step()

        stats = TrainingStats(
            losses={"loss_critic": loss_critic.item()},
            grad_norms={"critic": critic_grad},
        )

        if (
            self._optimization_step >= self.config.critic_warmup_steps
            and self._optimization_step % self.config.policy_update_freq == 0
        ):
            loss_actor = self._compute_loss_actor(fb)
            self.optimizers["actor"].zero_grad()
            loss_actor.backward()
            actor_grad = torch.nn.utils.clip_grad_norm_(
                self.policy.actor.parameters(), max_norm=clip
            ).item()
            self.optimizers["actor"].step()

            stats.losses["loss_actor"] = loss_actor.item()
            stats.grad_norms["actor"] = actor_grad

        self._update_target_networks()
        self._optimization_step += 1
        return stats

    def _compute_loss_critic(self, batch: dict[str, Any]) -> Tensor:
        observations = batch["state"]
        actions = batch[ACTION]
        observation_features = batch.get("observation_feature")
        rewards = batch["reward"]
        next_observations = batch["next_state"]
        done = batch["done"]
        next_observation_features = batch.get("next_observation_feature")

        with torch.no_grad():
            # Target action: target-actor mean plus target-policy-smoothing
            # noise (ResFiT samples a truncated Gaussian; adding a hard-clipped
            # Gaussian to the residual and clamping the combined action is
            # equivalent). The combine with the next base action and the clamp
            # to [-1, 1] happen inside the actor module.
            next_combined, _ = self.actor_target(
                next_observations,
                next_observation_features,
                add_noise=self.config.target_action_noise,
                stddev=self.config.target_action_noise_stddev,
                noise_clip=self.config.target_action_noise_clip,
            )
            if self.policy.config.num_discrete_actions is not None:
                next_combined = next_combined[..., :DISCRETE_DIMENSION_INDEX]

            # REDQ: minimum over a random subset of target critics
            q_targets = self.critic_target(
                observations=next_observations,
                actions=next_combined,
                observation_features=next_observation_features,
            )
            if self.config.num_subsample_critics is not None:
                indices = torch.randperm(self.config.num_critics)
                indices = indices[: self.config.num_subsample_critics]
                q_targets = q_targets[indices]
            min_q, _ = q_targets.min(dim=0)

            td_target = rewards + (1 - done) * self.config.discount * min_q
            if self.config.clip_q_target_to_reward_range:
                # Sparse success/failure rewards live in [0, 1]
                td_target = torch.clamp(td_target, min=0.0, max=1.0)

        # NOTE: We only want the continuous action part. In the buffer we have
        # the full action space (continuous + discrete).
        if self.policy.config.num_discrete_actions is not None:
            actions = actions[..., :DISCRETE_DIMENSION_INDEX]
        q_preds = self.critic_ensemble(
            observations=observations,
            actions=actions,
            observation_features=observation_features,
        )

        td_target_duplicate = einops.repeat(td_target, "b -> e b", e=q_preds.shape[0])
        critics_loss = (
            F.mse_loss(
                input=q_preds,
                target=td_target_duplicate,
                reduction="none",
            ).mean(dim=1)
        ).sum()
        return critics_loss

    def _compute_loss_actor(self, batch: dict[str, Any]) -> Tensor:
        observations = batch["state"]
        observation_features = batch.get("observation_feature")

        # Deterministic residual (no exploration noise) combined with the base action
        combined, residual = self.policy.actor(observations, observation_features)
        if self.policy.config.num_discrete_actions is not None:
            combined = combined[..., :DISCRETE_DIMENSION_INDEX]

        q_preds = self.critic_ensemble(
            observations=observations,
            actions=combined,
            observation_features=observation_features,
        )
        if self.config.policy_gradient_type == "ensemble_mean":
            q = q_preds.mean(dim=0)
        elif self.config.policy_gradient_type == "min_random_pair":
            indices = torch.randperm(self.config.num_critics)
            q = q_preds[indices[:2]].min(dim=0)[0]
        else:  # "q1"
            q = q_preds[0]

        actor_loss = -q.mean()
        if self.config.actor_l2_reg_weight > 0.0:
            actor_loss = actor_loss + self.config.actor_l2_reg_weight * torch.mean(
                torch.sum(residual**2, dim=-1)
            )
        return actor_loss

    def _update_target_networks(self) -> None:
        """EMA-update target critics and target actor head.

        The encoders are shared module objects between online and target
        modules, so their (identical) parameters are a no-op in the zip below;
        the actor-target loop additionally skips ``encoder.*`` keys explicitly.
        """
        tau = self.config.critic_target_update_weight
        for target_p, p in zip(
            self.critic_target.parameters(), self.critic_ensemble.parameters(), strict=True
        ):
            target_p.data.copy_(p.data * tau + target_p.data * (1.0 - tau))

        for (name, target_p), (_, p) in zip(
            self.actor_target.named_parameters(),
            self.policy.actor.named_parameters(),
            strict=True,
        ):
            if name.startswith("encoder."):
                continue
            target_p.data.copy_(p.data * tau + target_p.data * (1.0 - tau))

    def _prepare_forward_batch(
        self, batch: BatchType, *, include_complementary_info: bool = True
    ) -> dict[str, Any]:
        observations = batch["state"]
        next_observations = batch["next_state"]
        observation_features, next_observation_features = self.get_observation_features(
            observations, next_observations
        )
        forward_batch: dict[str, Any] = {
            ACTION: batch[ACTION],
            "reward": batch["reward"],
            "state": observations,
            "next_state": next_observations,
            "done": batch["done"],
            "observation_feature": observation_features,
            "next_observation_feature": next_observation_features,
        }
        if include_complementary_info and "complementary_info" in batch:
            forward_batch["complementary_info"] = batch["complementary_info"]
        return forward_batch

    def make_optimizers_and_scheduler(self) -> dict[str, Optimizer]:
        """
        Creates Adam optimizers for the actor and critic ensemble components.

        NOTE: as in SAC, the (shared) encoder parameters are trained through
        the critic optimizer and excluded from the actor optimizer.
        """
        actor_params = self.policy.get_optim_params()["actor"]
        self.optimizers = {
            "actor": torch.optim.Adam(actor_params, lr=self.config.actor_lr),
            "critic": torch.optim.Adam(self.critic_ensemble.parameters(), lr=self.config.critic_lr),
        }
        return self.optimizers

    def get_optimizers(self) -> dict[str, Optimizer]:
        return self.optimizers

    def get_weights(self) -> dict[str, Any]:
        """Send the actor state dict (incl. shared encoder) to the actor process."""
        return {
            "policy": move_state_dict_to_device(self.policy.actor.state_dict(), device="cpu"),
        }

    def load_weights(self, weights: dict[str, Any], device: str | torch.device = "cpu") -> None:
        """Load actor weights into the policy."""
        actor_sd = move_state_dict_to_device(weights["policy"], device=device)
        self.policy.actor.load_state_dict(actor_sd)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Algorithm-owned trainable tensors.

        Encoder weights are stripped because they are owned by the policy
        (``policy.encoder_critic`` / ``policy.encoder_actor``) and already saved
        via ``policy.save_pretrained``.
        """
        bundle: dict[str, torch.Tensor] = {}
        for k, v in _strip_encoder_keys(self.critic_ensemble.state_dict()).items():
            bundle[f"critic_ensemble.{k}"] = v
        for k, v in _strip_encoder_keys(self.critic_target.state_dict()).items():
            bundle[f"critic_target.{k}"] = v
        for k, v in _strip_encoder_keys(self.actor_target.state_dict()).items():
            bundle[f"actor_target.{k}"] = v
        return bundle

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        device: str | torch.device = "cpu",
    ) -> None:
        """In-place load of algorithm-owned tensors."""
        critic_ensemble_state = _split_prefix(state_dict, "critic_ensemble.")
        critic_target_state = _split_prefix(state_dict, "critic_target.")
        self.critic_ensemble.load_state_dict(critic_ensemble_state, strict=False)
        self.critic_target.load_state_dict(critic_target_state, strict=False)

        actor_target_state = _split_prefix(state_dict, "actor_target.")
        self.actor_target.load_state_dict(actor_target_state, strict=False)

    def get_observation_features(
        self, observations: dict[str, Tensor], next_observations: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor] | None, dict[str, Tensor] | None]:
        """
        Get observation features from the policy encoder. It acts as a cache
        for the observation features. When the vision encoder is frozen, the
        observation features are not updated and we can save compute by
        caching them.
        """
        if self.policy.config.vision_encoder_name is None or not self.policy.config.freeze_vision_encoder:
            return None, None

        with torch.no_grad():
            observation_features = self.policy.actor.encoder.get_cached_image_features(observations)
            next_observation_features = self.policy.actor.encoder.get_cached_image_features(next_observations)

        return observation_features, next_observation_features


def _strip_encoder_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop ``encoder.*`` keys from a module state dict."""
    return {k: v for k, v in state.items() if not k.startswith("encoder.")}


def _split_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """Return the subset of ``state`` whose keys start with ``prefix``, prefix-stripped."""
    return {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
