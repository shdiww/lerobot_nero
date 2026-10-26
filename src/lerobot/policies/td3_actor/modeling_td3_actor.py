#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team.
# All rights reserved.
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

from dataclasses import asdict

import torch
import torch.nn as nn
from torch import Tensor

from lerobot.utils.constants import ACTION

from ..gaussian_actor.modeling_gaussian_actor import (
    DISCRETE_DIMENSION_INDEX,
    MLP,
    GaussianActorObservationEncoder,
    orthogonal_init,
)
from ..pretrained import PreTrainedPolicy
from .configuration_td3_actor import OBS_BASE_ACTION, TD3ActorConfig


class TD3ActorPolicy(PreTrainedPolicy):
    """Deterministic residual actor for TD3-style fine-tuning of a frozen base policy.

    Rollouts: ``select_action`` returns the **combined** action
    ``clamp(base_action + residual, -1, 1)`` (continuous part) with the
    discrete dimension passed through from the base action, so the replay
    buffer stores combined actions. Training: the TD3 algorithm calls the
    actor module directly for the policy-gradient loss.
    """

    config_class = TD3ActorConfig
    name = "td3_actor"

    def __init__(
        self,
        config: TD3ActorConfig | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        continuous_action_dim = config.output_features[ACTION].shape[0]
        if config.num_discrete_actions is not None:
            continuous_action_dim -= 1
        self._init_encoders()
        self._init_actor(continuous_action_dim)

        # Rollout exploration state (actor-process side). ``add_exploration_noise``
        # and ``exploration_stddev`` can be adjusted at runtime, e.g. to follow a
        # stddev schedule or to evaluate the policy deterministically.
        self.add_exploration_noise = True
        self.exploration_stddev = config.exploration_stddev
        self.exploration_noise_clip = config.exploration_noise_clip

    def get_optim_params(self) -> dict:
        optim_params = {
            "actor": [
                p
                for n, p in self.actor.named_parameters()
                if not n.startswith("encoder") or not self.shared_encoder
            ],
        }
        return optim_params

    def reset(self):
        """Reset the policy"""
        pass

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        raise NotImplementedError(
            "TD3ActorPolicy does not support action chunking. It returns single actions!"
        )

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select combined action for inference/rollout.

        Adds Gaussian exploration noise to the residual when
        ``add_exploration_noise`` is enabled and ``exploration_stddev`` > 0
        (e.g. ``eval_policy`` disables it to evaluate deterministically).
        """

        observations_features = None
        if self.shared_encoder and self.actor.encoder.has_images:
            observations_features = self.actor.encoder.get_cached_image_features(batch)

        combined, _ = self.actor(
            batch,
            observations_features,
            add_noise=self.add_exploration_noise and self.exploration_stddev > 0.0,
            stddev=self.exploration_stddev,
            noise_clip=self.exploration_noise_clip,
        )
        return combined

    def forward(self, batch: dict[str, Tensor | dict[str, Tensor]]) -> dict[str, Tensor]:
        """Actor forward pass (no exploration noise).

        Args:
            batch: A flat observation dict, or a training dict containing
                ``"state"`` (observations) and optionally ``"observation_feature"``
                (pre-computed encoder features).

        Returns:
            Dict with ``"action"`` (combined) and ``"residual_action"`` tensors.
        """
        observations = batch.get("state", batch)
        observation_features = batch.get("observation_feature") if isinstance(batch, dict) else None
        combined, residual = self.actor(observations, observation_features)
        return {"action": combined, "residual_action": residual}

    def _init_encoders(self):
        """Initialize shared or separate encoders for actor and critic."""
        self.shared_encoder = self.config.shared_encoder
        self.encoder_critic = GaussianActorObservationEncoder(self.config)
        self.encoder_actor = (
            self.encoder_critic if self.shared_encoder else GaussianActorObservationEncoder(self.config)
        )

    def _init_actor(self, continuous_action_dim):
        """Initialize the residual actor network."""
        # The residual actor concatenates the base action to the encoded
        # observation, so the trunk input grows by the continuous action dim.
        trunk_input_dim = self.encoder_actor.output_dim
        if self.config.residual:
            trunk_input_dim += continuous_action_dim
        self.actor = TD3ResidualActor(
            encoder=self.encoder_actor,
            network=MLP(
                input_dim=trunk_input_dim, **asdict(self.config.actor_network_kwargs)
            ),
            continuous_action_dim=continuous_action_dim,
            num_discrete_actions=self.config.num_discrete_actions,
            residual=self.config.residual,
            action_scale=self.config.action_scale,
            init_final=self.config.actor_last_layer_init_scale,
            encoder_is_shared=self.shared_encoder,
        )


class TD3ResidualActor(nn.Module):
    """Deterministic residual actor head on top of the observation encoder.

    forward returns ``(combined_action, residual_action)``. In residual mode the
    base action (``observation.base_action``, normalized action space) is
    concatenated to the encoded observation, the residual is bounded by
    ``tanh * action_scale`` and the combined action is clamped to [-1, 1] to
    match environment execution. The discrete action dimension (if any) is
    passed through from the base action untouched.
    """

    def __init__(
        self,
        encoder: GaussianActorObservationEncoder,
        network: nn.Module,
        continuous_action_dim: int,
        num_discrete_actions: int | None = None,
        residual: bool = True,
        action_scale: float = 0.2,
        init_final: float | None = None,
        encoder_is_shared: bool = False,
    ):
        super().__init__()
        self.encoder: GaussianActorObservationEncoder = encoder
        self.network = network
        self.continuous_action_dim = continuous_action_dim
        self.num_discrete_actions = num_discrete_actions
        self.residual = residual
        self.action_scale = action_scale if residual else 1.0
        self.encoder_is_shared = encoder_is_shared

        # Find the last Linear layer's output dimension
        for layer in reversed(network.net):
            if isinstance(layer, nn.Linear):
                out_features = layer.out_features
                break
        self.mean_layer = nn.Linear(out_features, continuous_action_dim)
        if init_final is not None:
            nn.init.uniform_(self.mean_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.mean_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.mean_layer.weight)

    def forward(
        self,
        observations: dict[str, Tensor],
        observation_features: dict[str, Tensor] | None = None,
        *,
        add_noise: bool = False,
        stddev: float = 0.0,
        noise_clip: float = 0.3,
    ) -> tuple[Tensor, Tensor]:
        """Compute the (combined, residual) action pair.

        Args:
            observations: Flat observation dict. Must contain
                ``observation.base_action`` in residual mode.
            observation_features: Optional pre-computed image features cache.
            add_noise: Add Gaussian exploration noise to the residual (rollout
                mode only; training uses the deterministic mean).
            stddev: Exploration noise stddev (normalized action space).
            noise_clip: Hard clip on the exploration noise sample.
        """
        # We detach the encoder if it is shared to avoid backprop through it
        # (the encoder is trained by the critic optimizer).
        obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)

        if self.residual:
            base_action = observations[OBS_BASE_ACTION]
            base_continuous = (
                base_action[..., :DISCRETE_DIMENSION_INDEX]
                if self.num_discrete_actions is not None
                else base_action
            )
            actor_input = torch.cat([obs_enc, base_continuous], dim=-1)
        else:
            base_continuous = None
            actor_input = obs_enc

        residual = torch.tanh(self.mean_layer(self.network(actor_input))) * self.action_scale

        if add_noise and stddev > 0.0:
            noise = (torch.randn_like(residual) * stddev).clamp(-noise_clip, noise_clip)
            residual = residual + noise

        if self.residual:
            combined_continuous = torch.clamp(base_continuous + residual, -1.0, 1.0)
            if self.num_discrete_actions is not None:
                combined = torch.cat(
                    [combined_continuous, base_action[..., DISCRETE_DIMENSION_INDEX:]], dim=-1
                )
            else:
                combined = combined_continuous
        else:
            combined = residual

        return combined, residual
