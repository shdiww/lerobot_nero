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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import CriticNetworkConfig

from ..configs import RLAlgorithmConfig

TD3_POLICY_GRADIENT_TYPES = ("ensemble_mean", "min_random_pair", "q1")


@RLAlgorithmConfig.register_subclass("td3")
@dataclass
class TD3AlgorithmConfig(RLAlgorithmConfig):
    """(Residual) TD3 algorithm configuration with a REDQ critic ensemble.

    Port of the ResFiT update rule (amazon-far/residual-offpolicy-rl,
    arXiv 2509.19301): deterministic delayed-policy-updates actor on top of an
    ensemble of Q critics. With ``num_critics > 2`` and
    ``policy_gradient_type="ensemble_mean"`` this is the REDQ-style variant
    used in the paper; ``num_critics=2, policy_gradient_type="q1"`` recovers
    standard TD3. The policy-side (residual actor + observation encoder) lives
    in :class:`~lerobot.policies.td3_actor.TD3ActorConfig` and is referenced
    via :attr:`policy_config`.

    Defaults follow the ResFiT paper runs (e.g. cansorting): actor_lr=1e-6,
    critic_lr=1e-4, tau=0.005, critic warmup 10k steps, delayed actor updates.
    """

    # Optimizer learning rates
    # Learning rate for the actor network (small: large steps corrupt the base policy)
    actor_lr: float = 1e-6
    # Learning rate for the critic ensemble
    critic_lr: float = 1e-4

    # Bellman update
    # Discount factor
    discount: float = 0.99
    # Weight for the target-network EMA update
    critic_target_update_weight: float = 0.005

    # Critic ensemble (REDQ)
    # Number of critics in the ensemble (10 = REDQ, 2 = standard TD3)
    num_critics: int = 10
    # Number of randomly subsampled critics whose minimum forms the Bellman
    # target (REDQ uses 2)
    num_subsample_critics: int = 2
    # How the policy gradient aggregates the ensemble: "ensemble_mean" (mean
    # over all heads, standard REDQ), "min_random_pair" (min over 2 random
    # heads), "q1" (first head only, standard TD3)
    policy_gradient_type: str = "ensemble_mean"
    # Configuration for the critic network architecture
    critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)

    # Update loop
    # Update-to-data ratio. Set to >1 for extra critic updates per env step
    # (ResFiT paper runs use 4).
    utd_ratio: int = 1
    # Delayed policy updates: the actor is updated every
    # ``policy_update_freq``-th optimization step (TD3)
    policy_update_freq: int = 2
    # Critic-only warmup: actor updates start after this many optimization steps
    critic_warmup_steps: int = 10_000
    # Gradient clipping norm
    grad_clip_norm: float = 40.0

    # Target policy smoothing (TD3). ResFiT samples the target action from a
    # truncated Gaussian around the target actor mean instead of adding
    # clipped noise; with a symmetric clip this is equivalent in expectation.
    # Whether to add noise to the target action
    target_action_noise: bool = True
    # Stddev of the target-action noise (normalized action space)
    target_action_noise_stddev: float = 0.05
    # Hard clip on the target-action noise sample
    target_action_noise_clip: float = 0.3

    # Clamp the Bellman target to [0, 1] (sparse success/failure rewards)
    clip_q_target_to_reward_range: bool = True

    # L2 regularization on the residual action magnitude (ResFiT
    # ``actor.action_l2_reg_weight``; 0 disables)
    actor_l2_reg_weight: float = 0.0

    # Optimizations
    # torch.compile is currently disabled by default
    use_torch_compile: bool = False

    # Policy config
    policy_config: PreTrainedConfig | None = None

    @classmethod
    def from_policy_config(cls, policy_cfg: PreTrainedConfig) -> TD3AlgorithmConfig:
        """Build an algorithm config with default hyperparameters for a given policy."""
        return cls(policy_config=policy_cfg)
