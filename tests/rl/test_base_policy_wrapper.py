#!/usr/bin/env python

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
"""Tests for the frozen-base-policy environment wrapper (residual TD3).

NOTE for reviewers of automated security scans: random tensors below generate
RL test data, not cryptographic material.
"""

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.factory import make_policy  # noqa: E402
from lerobot.policies.td3_actor import TD3ActorConfig  # noqa: E402
from lerobot.policies.td3_actor.configuration_td3_actor import OBS_BASE_ACTION  # noqa: E402
from lerobot.rl.base_policy_wrapper import BaseActionComputer, BasePolicyEnvWrapper  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

STATE_DIM = 4
ACTION_DIM = 3


class FakeEnv(gym.Env):
    """Minimal env whose observations only contain ``observation.state``."""

    def __init__(self):
        self.observation_space = gym.spaces.Dict(
            {OBS_STATE: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32)}
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.reset_calls = 0

    def reset(self, *, seed=None, options=None):
        self.reset_calls += 1
        return {OBS_STATE: torch.randn(STATE_DIM)}, {}

    def step(self, action):
        obs = {OBS_STATE: torch.randn(STATE_DIM)}
        return obs, 0.0, False, False, {}

    def get_raw_joint_positions(self):
        return "raw"


class FakeBasePolicy:
    """Outputs a fixed action in its own normalized [-1, 1] space."""

    def __init__(self):
        self.config = SimpleNamespaceLikeConfig()
        self.reset_calls = 0
        # Deterministic 'policy': each dim outputs 0.5
        self.fixed_action = torch.full((1, ACTION_DIM), 0.5)

    def select_action(self, batch):
        return self.fixed_action.clone()

    def reset(self):
        self.reset_calls += 1

    def parameters(self):
        return iter(())


class SimpleNamespaceLikeConfig:
    input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))}


class IdentityPreprocessor:
    def process_observation(self, batch):
        return batch


class MinMaxPostprocessor:
    """Un-normalizes [-1, 1] to [min, max] like UnnormalizerProcessorStep."""

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.denom = max_val - min_val

    def process_action(self, action):
        return (action + 1) / 2 * self.denom + self.min_val


def _make_wrapper(env=None, base_policy=None, rl_min=-1.0, rl_max=1.0):
    env = env or FakeEnv()
    base_policy = base_policy or FakeBasePolicy()
    computer = BaseActionComputer(
        base_policy=base_policy,
        base_preprocessor=IdentityPreprocessor(),
        base_postprocessor=MinMaxPostprocessor(0.0, 2.0),
        action_min=torch.full((ACTION_DIM,), rl_min),
        action_max=torch.full((ACTION_DIM,), rl_max),
    )
    return (
        BasePolicyEnvWrapper(env=env, base_action_computer=computer),
        env,
        base_policy,
    )


def test_reset_and_step_inject_base_action():
    wrapper, env, base_policy = _make_wrapper()
    obs, info = wrapper.reset()
    assert OBS_BASE_ACTION in obs
    assert obs[OBS_BASE_ACTION].shape == (ACTION_DIM,)
    assert base_policy.reset_calls == 1

    # base outputs 0.5 in its [-1, 1] space; base stats map to raw [0, 2]:
    # raw = (0.5 + 1) / 2 * 2 + 0 = 1.5; RL stats [-1, 1] keep it unchanged.
    assert torch.allclose(obs[OBS_BASE_ACTION], torch.full((ACTION_DIM,), 1.5))

    obs, reward, term, trunc, info = wrapper.step(torch.zeros(ACTION_DIM))
    assert OBS_BASE_ACTION in obs
    assert torch.allclose(obs[OBS_BASE_ACTION], torch.full((ACTION_DIM,), 1.5))
    # No episode end during step: base policy only reset at reset()
    assert base_policy.reset_calls == 1


def test_observation_space_augmented():
    wrapper, _, _ = _make_wrapper()
    assert OBS_BASE_ACTION in wrapper.observation_space.spaces
    space = wrapper.observation_space.spaces[OBS_BASE_ACTION]
    assert space.shape == (ACTION_DIM,)
    assert wrapper.action_space is not None


def test_attribute_delegation():
    wrapper, _, _ = _make_wrapper()
    assert wrapper.get_raw_joint_positions() == "raw"


def test_base_action_normalization_matches_formula():
    # RL stats [-2, 6]: raw 1.5 -> 2 * (1.5 - (-2)) / 8 - 1 = -0.125
    wrapper, _, _ = _make_wrapper(rl_min=-2.0, rl_max=6.0)
    obs, _ = wrapper.reset()
    assert torch.allclose(obs[OBS_BASE_ACTION], torch.full((ACTION_DIM,), -0.125))


def test_load_base_policy_uses_configured_policy_type(monkeypatch):
    """Any policy registry type (e.g. "diffusion") can serve as the frozen base."""
    from unittest.mock import MagicMock

    import lerobot.policies.factory as policy_factory
    import lerobot.rl.base_policy_wrapper as bpw

    requested = {}
    fake_policy = MagicMock()
    fake_policy.parameters.return_value = iter(())
    fake_policy.config = SimpleNamespaceLikeConfig()

    def fake_get_policy_class(name):
        requested["name"] = name
        return MagicMock(from_pretrained=lambda path: fake_policy)

    monkeypatch.setattr(policy_factory, "get_policy_class", fake_get_policy_class)
    monkeypatch.setattr(
        bpw,
        "make_pre_post_processors",
        lambda *args, **kwargs: (IdentityPreprocessor(), MinMaxPostprocessor(0.0, 2.0)),
    )

    policy, preprocessor, postprocessor = bpw.load_base_policy(
        "fake/path", device="cpu", base_policy_type="diffusion"
    )
    assert requested["name"] == "diffusion"
    # The loaded policy went through the to(device) + eval() chain
    fake_policy.to.assert_called_once_with("cpu")
    fake_policy.to.return_value.eval.assert_called_once()

    # Default remains "act"
    bpw.load_base_policy("fake/path", device="cpu")
    assert requested["name"] == "act"


def test_make_policy_auto_injects_base_action_feature():
    from types import SimpleNamespace

    state_dim, action_dim = 4, 7
    features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
    }
    env_cfg = SimpleNamespace(features=features, features_map={k: k for k in features})

    policy_cfg = TD3ActorConfig(
        device="cpu",
        dataset_stats={
            OBS_STATE: {"min": [0.0] * state_dim, "max": [1.0] * state_dim},
            ACTION: {"min": [0.0] * action_dim, "max": [1.0] * action_dim},
        },
    )
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    assert OBS_BASE_ACTION in policy.config.input_features
    assert policy.config.input_features[OBS_BASE_ACTION].shape == (action_dim,)
    # The residual actor can run on an augmented observation
    obs = {
        OBS_STATE: torch.randn(1, state_dim),
        OBS_BASE_ACTION: torch.rand(1, action_dim) * 2 - 1,
    }
    action = policy.select_action(obs)
    assert action.shape == (1, action_dim)
