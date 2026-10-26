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

"""Environment wrapper exposing a frozen base policy through observations.

Port of ResFiT's ``BasePolicyVecEnvWrapper`` (amazon-far/residual-offpolicy-rl)
to this fork's actor rollout loop: instead of combining actions inside the
environment, this wrapper only *observes* — it runs the frozen base policy
(e.g. ACT) on every observation and injects the result as
``observation.base_action`` in the normalized [-1, 1] action space. The
residual actor policy (``td3_actor``) combines base + residual itself and the
replay buffer stores combined actions.
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.td3_actor.configuration_td3_actor import OBS_BASE_ACTION
from lerobot.utils.constants import ACTION


def load_base_policy(base_policy_path: str, device: str):
    """Load a frozen base policy (e.g. ACT) plus its pre/post processors.

    Processor configs saved alongside the checkpoint take precedence; otherwise
    the processors are rebuilt from the dataset stats stored in the policy
    config.
    """
    from lerobot.policies.factory import get_policy_class

    policy_cls = get_policy_class("act")
    base_policy = policy_cls.from_pretrained(base_policy_path)
    base_policy = base_policy.to(device).eval()
    for param in base_policy.parameters():
        param.requires_grad = False

    try:
        base_preprocessor, base_postprocessor = make_pre_post_processors(
            base_policy.config, pretrained_path=base_policy_path
        )
    except Exception:
        logging.warning(
            f"Could not load processor configs from '{base_policy_path}'; rebuilding "
            "them from the policy config's dataset stats."
        )
        base_preprocessor, base_postprocessor = make_pre_post_processors(
            base_policy.config, dataset_stats=base_policy.config.dataset_stats
        )
    return base_policy, base_preprocessor, base_postprocessor


class BaseActionComputer:
    """Runs a frozen base policy on observations and normalizes its action.

    Shared by the environment wrapper (rollouts) and the offline-buffer
    labeling pass (demonstrations): both must map a raw observation through the
    base policy's own pre/post processors into the raw action space, then
    re-normalize into the RL policy's [-1, 1] action space using the same
    ``action`` min/max dataset stats as the residual policy's post-processor.
    """

    def __init__(
        self,
        base_policy: Any,
        base_preprocessor: Any,
        base_postprocessor: Any,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
        eps: float = 1e-8,
    ):
        self.base_policy = base_policy
        self.base_preprocessor = base_preprocessor
        self.base_postprocessor = base_postprocessor
        self.eps = eps

        self._action_min = torch.as_tensor(action_min, dtype=torch.float32)
        self._action_max = torch.as_tensor(action_max, dtype=torch.float32)
        denom = self._action_max - self._action_min
        # Match NormalizerProcessorStep: keep division well-defined when a
        # dimension has zero range.
        denom = torch.where(denom == 0, torch.tensor(self.eps, dtype=torch.float32), denom)
        self._action_denom = denom
        self.action_dim = int(self._action_min.shape[0])

    def __call__(self, raw_obs: dict) -> torch.Tensor:
        """Return the base action, shape (action_dim,), in the [-1, 1] space.

        ``raw_obs`` is an un-batched observation dict (single frame, raw
        values); the base policy's preprocessor adds the batch dimension.
        """
        batch = {k: v for k, v in raw_obs.items() if k in self.base_policy.config.input_features}
        normalized = self.base_preprocessor.process_observation(batch)
        with torch.no_grad():
            action = self.base_policy.select_action(batch=normalized)
        # Un-normalize with the base policy's stats back to the raw action
        # space, then re-normalize with the RL policy's action stats.
        raw_action = self.base_postprocessor.process_action(action)
        raw_action = raw_action.detach().to(torch.float32).squeeze(0)
        return 2 * (raw_action - self._action_min) / self._action_denom - 1

    def reset(self):
        """Reset the (chunked) base policy, e.g. at episode boundaries."""
        self.base_policy.reset()


def _validate_action_stats(dataset_stats: dict, action_dim: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract and validate the action min/max stats used for base actions."""
    action_stats = (dataset_stats or {}).get(ACTION)
    if action_stats is None or "min" not in action_stats or "max" not in action_stats:
        raise ValueError(
            "Residual TD3 training requires action min/max dataset statistics "
            "(policy_cfg.dataset_stats['action']) to normalize base actions; they must "
            "be the same stats used by the residual policy's post-processor."
        )
    action_min = torch.tensor(action_stats["min"], dtype=torch.float32)
    action_max = torch.tensor(action_stats["max"], dtype=torch.float32)
    if action_dim is not None and action_min.shape[0] != action_dim:
        raise ValueError(
            f"Action stats dimension ({action_min.shape[0]}) does not match the policy "
            f"action dimension ({action_dim})"
        )
    return action_min, action_max


class BasePolicyEnvWrapper:
    """Wraps a robot env with a frozen base policy for residual RL training.

    On every ``reset``/``step`` this wrapper runs the frozen base policy on the
    raw observation and stores the resulting normalized action under
    ``observation.base_action``. The base policy is reset when an episode
    terminates (ACT is a chunked/temporal policy and must not carry state
    across episodes).

    Actions are *not* combined here: the residual actor policy returns the
    combined action, so the standard actor loop stores combined actions in the
    replay buffer unchanged.
    """

    def __init__(
        self,
        env: gym.Env,
        base_action_computer: BaseActionComputer,
        base_action_key: str = OBS_BASE_ACTION,
    ):
        self.env = env
        self.computer = base_action_computer
        self.base_action_key = base_action_key
        self.action_dim = base_action_computer.action_dim

        # Expose the augmented observation space so downstream feature
        # derivation sees ``observation.base_action``.
        spaces = dict(env.observation_space.spaces)
        spaces[self.base_action_key] = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(spaces)
        self.action_space = env.action_space

    def _augment(self, raw_obs: dict) -> dict:
        obs = dict(raw_obs)
        obs[self.base_action_key] = self.computer(obs)
        return obs

    def reset(self, **kwargs) -> tuple[dict, dict]:
        raw_obs, info = self.env.reset(**kwargs)
        self.computer.reset()
        return self._augment(raw_obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self.computer.reset()
        return self._augment(obs), reward, terminated, truncated, info

    def close(self):
        return self.env.close()

    def render(self):
        return self.env.render()

    def __getattr__(self, name: str):
        # Delegate unknown attributes (e.g. ``get_raw_joint_positions``) to the
        # underlying environment.
        return getattr(self.env, name)


def _make_computer_from_base_policy_path(
    base_policy_path: str,
    dataset_stats: dict,
    device: str,
    action_dim: int | None = None,
) -> BaseActionComputer:
    action_min, action_max = _validate_action_stats(dataset_stats, action_dim=action_dim)
    base_policy, base_preprocessor, base_postprocessor = load_base_policy(base_policy_path, device=device)
    return BaseActionComputer(
        base_policy=base_policy,
        base_preprocessor=base_preprocessor,
        base_postprocessor=base_postprocessor,
        action_min=action_min,
        action_max=action_max,
    )


def wrap_env_with_base_policy(env: gym.Env, policy_cfg, device: str) -> BasePolicyEnvWrapper:
    """Wrap ``env`` with its frozen base policy according to ``policy_cfg``.

    ``policy_cfg`` must be a ``TD3ActorConfig`` with ``residual=True`` and a
    ``base_policy_path``. The action min/max stats come from
    ``policy_cfg.dataset_stats['action']`` and must match the stats used to
    build the residual policy's post-processor.
    """
    if not getattr(policy_cfg, "residual", False) or policy_cfg.base_policy_path is None:
        raise ValueError(
            "wrap_env_with_base_policy requires a residual TD3ActorConfig with base_policy_path set"
        )
    computer = _make_computer_from_base_policy_path(
        policy_cfg.base_policy_path,
        dataset_stats=policy_cfg.dataset_stats,
        device=device,
        action_dim=policy_cfg.output_features[ACTION].shape[0],
    )
    logging.info(f"Wrapping env with frozen base policy from '{policy_cfg.base_policy_path}'")
    return BasePolicyEnvWrapper(env=env, base_action_computer=computer)


def make_base_action_labeler(
    base_policy_path: str,
    dataset_stats: dict,
    device: str,
) -> BaseActionComputer:
    """Build a per-frame base-action labeler for demonstration datasets.

    The returned object is callable on raw dataset samples (returning a
    (action_dim,) tensor in the normalized [-1, 1] action space) and exposes
    ``reset()`` so ``ReplayBuffer.from_lerobot_dataset`` can reset chunked base
    policies at episode boundaries.
    """
    return _make_computer_from_base_policy_path(base_policy_path, dataset_stats=dataset_stats, device=device)
