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
"""Tests for the (residual) TD3 actor policy and TD3+REDQ algorithm implementation.

NOTE for reviewers of automated security scans: ``torch.rand`` / ``torch.randn``
below generate RL test tensors, not cryptographic material.
"""

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import torch  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_policy_config  # noqa: E402
from lerobot.policies.gaussian_actor import GaussianActorConfig  # noqa: E402
from lerobot.policies.gaussian_actor.modeling_gaussian_actor import GaussianActorPolicy  # noqa: E402
from lerobot.policies.td3_actor import TD3ActorConfig  # noqa: E402
from lerobot.policies.td3_actor.modeling_td3_actor import TD3ActorPolicy  # noqa: E402
from lerobot.rl.algorithms.factory import (  # noqa: E402
    get_algorithm_class,
    make_algorithm,
    make_algorithm_config,
)
from lerobot.rl.algorithms.td3 import TD3Algorithm, TD3AlgorithmConfig  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

OBS_BASE_ACTION = "observation.base_action"


@pytest.fixture(autouse=True)
def set_random_seed():
    set_seed(42)


def _make_td3_config(
    state_dim: int = 10,
    continuous_action_dim: int = 6,
    num_discrete_actions: int | None = 1,
) -> TD3ActorConfig:
    action_dim = continuous_action_dim + (1 if num_discrete_actions is not None else 0)
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        OBS_BASE_ACTION: PolicyFeature(type=FeatureType.STATE, shape=(action_dim,)),
    }
    config = TD3ActorConfig(
        input_features=input_features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        dataset_stats={
            OBS_STATE: {"min": [0.0] * state_dim, "max": [1.0] * state_dim},
            ACTION: {"min": [0.0] * action_dim, "max": [1.0] * action_dim},
        },
        num_discrete_actions=num_discrete_actions,
    )
    config.validate_features()
    return config


def _make_algorithm(
    state_dim: int = 10,
    continuous_action_dim: int = 6,
    num_discrete_actions: int | None = 1,
    critic_warmup_steps: int = 0,
    policy_update_freq: int = 1,
) -> tuple[TD3Algorithm, TD3ActorPolicy]:
    policy_cfg = _make_td3_config(
        state_dim=state_dim,
        continuous_action_dim=continuous_action_dim,
        num_discrete_actions=num_discrete_actions,
    )
    policy = TD3ActorPolicy(config=policy_cfg)
    policy.train()
    algo_config = TD3AlgorithmConfig.from_policy_config(policy_cfg)
    algo_config.critic_warmup_steps = critic_warmup_steps
    algo_config.policy_update_freq = policy_update_freq
    algorithm = TD3Algorithm(policy=policy, config=algo_config)
    algorithm.make_optimizers_and_scheduler()
    return algorithm, policy


def _make_batch(
    batch_size: int = 4,
    state_dim: int = 10,
    action_dim: int = 7,
) -> dict:
    obs = {
        OBS_STATE: torch.randn(batch_size, state_dim),
        OBS_BASE_ACTION: torch.rand(batch_size, action_dim) * 2.0 - 1.0,
    }
    next_obs = {
        OBS_STATE: torch.randn(batch_size, state_dim),
        OBS_BASE_ACTION: torch.rand(batch_size, action_dim) * 2.0 - 1.0,
    }
    return {
        ACTION: torch.rand(batch_size, action_dim) * 2.0 - 1.0,
        "reward": torch.rand(batch_size),
        "state": obs,
        "next_state": next_obs,
        "done": torch.zeros(batch_size),
        "complementary_info": {},
    }


def _td3_batch_iterator(**batch_kwargs):
    """Infinite iterator yielding fresh batches (mirrors a real DataMixer iterator)."""
    while True:
        yield _make_batch(**batch_kwargs)


def test_policy_and_algorithm_factory_registration():
    assert get_policy_class("td3_actor") is TD3ActorPolicy
    assert get_algorithm_class("td3") is TD3Algorithm

    policy_cfg = make_policy_config("td3_actor")
    assert isinstance(policy_cfg, TD3ActorConfig)
    algo_cfg = make_algorithm_config("td3")
    assert isinstance(algo_cfg, TD3AlgorithmConfig)
    assert algo_cfg.type == "td3"


def test_make_algorithm_rejects_non_td3_policy():
    gaussian_policy = GaussianActorPolicy(
        config=GaussianActorConfig(
            input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,))},
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        )
    )
    algo_config = TD3AlgorithmConfig(policy_config=_make_td3_config())
    with pytest.raises(TypeError, match="TD3ActorPolicy"):
        make_algorithm(cfg=algo_config, policy=gaussian_policy)


def test_zero_init_select_action_equals_base_policy():
    """With actor_last_layer_init_scale=0.0 the initial residual is exactly zero,
    so the deterministic policy is identical to the frozen base policy."""
    algorithm, policy = _make_algorithm()
    assert policy.config.actor_last_layer_init_scale == 0.0

    obs = {
        OBS_STATE: torch.randn(2, 10),
        OBS_BASE_ACTION: torch.rand(2, 7) * 2.0 - 1.0,
    }
    policy.add_exploration_noise = False
    action = policy.select_action(obs)
    assert action.shape == (2, 7)
    # Continuous part clamped to [-1, 1]; discrete part passed through untouched
    assert torch.allclose(action, obs[OBS_BASE_ACTION])
    assert action.min() >= -1.0 and action.max() <= 1.0


def test_select_action_exploration_noise_perturbs_residual():
    algorithm, policy = _make_algorithm()
    obs = {
        OBS_STATE: torch.randn(2, 10),
        OBS_BASE_ACTION: torch.rand(2, 7) * 2.0 - 1.0,
    }
    policy.exploration_stddev = 0.1
    policy.add_exploration_noise = True
    noisy = policy.select_action(obs)
    policy.add_exploration_noise = False
    clean = policy.select_action(obs)
    # Zero-init residual means `clean` is the base action; noise moves off it
    assert not torch.allclose(noisy, clean)
    assert noisy.min() >= -1.0 and noisy.max() <= 1.0


def test_residual_bounded_by_action_scale():
    algorithm, policy = _make_algorithm()
    # Give the final layer non-zero weights so the residual is non-trivial
    with torch.no_grad():
        policy.actor.mean_layer.weight.uniform_(-1.0, 1.0)
        policy.actor.mean_layer.bias.uniform_(-1.0, 1.0)
    obs = {OBS_STATE: torch.randn(8, 10), OBS_BASE_ACTION: torch.zeros(8, 7)}
    combined, residual = policy.actor(obs)
    assert residual.shape == (8, 6)
    assert residual.abs().max() <= policy.config.action_scale + 1e-6
    assert combined.min() >= -1.0 and combined.max() <= 1.0
    # Discrete dimension is a passthrough of the (zero) base action
    assert torch.all(combined[:, -1] == 0.0)


def test_update_after_warmup_trains_actor_and_critics():
    algorithm, policy = _make_algorithm(critic_warmup_steps=0, policy_update_freq=1)
    actor_params_before = [p.clone() for p in policy.actor.mean_layer.parameters()]
    critic_params_before = [p.clone() for p in algorithm.critic_ensemble.parameters()]

    iterator = _td3_batch_iterator()
    stats = algorithm.update(iterator)

    assert "loss_critic" in stats.losses
    assert "loss_actor" in stats.losses
    actor_changed = any(
        not torch.allclose(p, p0)
        for p, p0 in zip(policy.actor.mean_layer.parameters(), actor_params_before, strict=True)
    )
    critic_changed = any(
        not torch.allclose(p, p0)
        for p, p0 in zip(algorithm.critic_ensemble.parameters(), critic_params_before, strict=True)
    )
    assert actor_changed
    assert critic_changed


def test_actor_frozen_during_critic_warmup():
    algorithm, policy = _make_algorithm(critic_warmup_steps=100, policy_update_freq=1)
    actor_params_before = [p.clone() for p in policy.actor.mean_layer.parameters()]

    iterator = _td3_batch_iterator()
    stats = algorithm.update(iterator)

    assert "loss_actor" not in stats.losses
    for p, p0 in zip(policy.actor.mean_layer.parameters(), actor_params_before, strict=True):
        assert torch.allclose(p, p0)


def test_delayed_actor_updates_follow_policy_update_freq():
    # policy_update_freq=2: actor updated on optimization steps 0, 2, 4, ...
    algorithm, policy = _make_algorithm(critic_warmup_steps=0, policy_update_freq=2)
    iterator = _td3_batch_iterator()
    stats = algorithm.update(iterator)  # step 0 -> actor updated
    assert "loss_actor" in stats.losses
    stats = algorithm.update(iterator)  # step 1 -> actor skipped
    assert "loss_actor" not in stats.losses
    stats = algorithm.update(iterator)  # step 2 -> actor updated
    assert "loss_actor" in stats.losses


def test_utd_ratio_consumes_multiple_batches():
    algorithm, policy = _make_algorithm(critic_warmup_steps=0, policy_update_freq=1)
    algorithm.config.utd_ratio = 3
    iterator = iter([_make_batch(), _make_batch(), _make_batch()])
    # update() must pull exactly utd_ratio batches: 3 pulls succeed, no more
    stats = algorithm.update(iterator)
    assert "loss_critic" in stats.losses
    with pytest.raises(StopIteration):
        next(iterator)


def test_q_target_clamped_to_reward_range():
    algorithm, policy = _make_algorithm()
    assert algorithm.config.clip_q_target_to_reward_range
    # The clamp is applied on the Bellman target inside _compute_loss_critic;
    # with a large reward the critic loss must still be finite
    batch = _make_batch()
    batch["reward"] = torch.full((batch["reward"].shape[0],), 5.0)
    batch["done"] = torch.ones(batch["reward"].shape[0])
    fb = algorithm._prepare_forward_batch(batch)
    loss = algorithm._compute_loss_critic(fb)
    assert torch.isfinite(loss)


def test_state_dict_round_trip():
    algorithm, policy = _make_algorithm(critic_warmup_steps=0, policy_update_freq=1)
    iterator = _td3_batch_iterator()
    for _ in range(3):
        algorithm.update(iterator)

    state = algorithm.state_dict()
    assert any(k.startswith("critic_ensemble.") for k in state)
    assert any(k.startswith("critic_target.") for k in state)
    assert any(k.startswith("actor_target.") for k in state)
    assert not any(k.startswith("actor_target.encoder.") for k in state)

    # Perturb, then restore: the algorithm must remain usable
    with torch.no_grad():
        for p in algorithm.critic_ensemble.parameters():
            p.add_(1.0)
    algorithm.load_state_dict(state)
    fb = algorithm._prepare_forward_batch(_make_batch())
    algorithm._compute_loss_critic(fb)  # must not raise


def test_get_weights_load_weights_round_trip():
    algorithm, policy = _make_algorithm(critic_warmup_steps=0, policy_update_freq=1)
    # On CPU, ``move_state_dict_to_device`` keeps tensor identity, so snapshot
    # explicitly (in production the learner is on GPU and the move copies).
    raw_weights = algorithm.get_weights()
    initial_weights = {"policy": {k: v.clone() for k, v in raw_weights["policy"].items()}}
    assert "policy" in initial_weights
    initial_head = {k: v.clone() for k, v in policy.actor.state_dict().items() if "mean_layer" in k}

    for _ in range(3):
        algorithm.update(_td3_batch_iterator())
    moved = any(
        not torch.allclose(policy.actor.state_dict()[k], v) for k, v in initial_head.items()
    )
    assert moved  # training moved the actor head away from zero-init

    algorithm.load_weights(initial_weights)
    for k, v in initial_head.items():
        assert torch.allclose(policy.actor.state_dict()[k], v)


def test_td3_algorithm_supports_continuous_only_actions():
    algorithm, policy = _make_algorithm(num_discrete_actions=None)
    obs = {OBS_STATE: torch.randn(2, 10), OBS_BASE_ACTION: torch.rand(2, 6) * 2.0 - 1.0}
    policy.add_exploration_noise = False
    action = policy.select_action(obs)
    assert action.shape == (2, 6)

    iterator = _td3_batch_iterator(action_dim=6)
    stats = algorithm.update(iterator)
    assert "loss_critic" in stats.losses
