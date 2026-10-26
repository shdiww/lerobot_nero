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

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import MultiAdamConfig
from lerobot.utils.constants import ACTION, OBS_STATE

from ..gaussian_actor.configuration_gaussian_actor import (
    ActorLearnerConfig,
    ActorNetworkConfig,
    ConcurrencyConfig,
    is_image_feature,
)

# Observation key holding the (frozen) base-policy action, in the normalized
# [-1, 1] action space. Injected by the environment-side base-policy wrapper
# during rollouts and by the offline-buffer labeling pass for demonstrations.
# The residual policy concatenates it with the encoded observation; it is
# intentionally NOT normalized by the policy preprocessor (it already lives in
# the normalized action space).
OBS_BASE_ACTION = "observation.base_action"


@PreTrainedConfig.register_subclass("td3_actor")
@dataclass
class TD3ActorConfig(PreTrainedConfig):
    """Deterministic (residual) actor configuration, as used by TD3.

    In residual mode the policy does not replace the base (e.g. ACT) policy:
    it predicts only a bounded correction ``action_scale * tanh(x)`` on top of
    the frozen base action exposed through ``observation.base_action``, and
    ``select_action`` returns the clamped combined action so that the replay
    buffer stores combined actions. With ``actor_last_layer_init_scale = 0.0``
    the residual starts at exactly zero, making the initial policy identical
    to the base policy. The critics, target networks and the TD3/REDQ update
    loop live on the algorithm side (see ``lerobot.rl.algorithms.td3``).

    CLI: ``--policy.type=td3_actor``.
    """

    # Mapping of feature types to normalization modes
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Statistics for normalizing different types of inputs
    dataset_stats: dict[str, dict[str, list[float]]] | None = field(
        default_factory=lambda: {
            "observation.image": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            OBS_STATE: {
                "min": [0.0, 0.0],
                "max": [1.0, 1.0],
            },
            ACTION: {
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 1.0],
            },
        }
    )

    # Architecture specifics
    # Device to run the model on (e.g. "cuda", "cpu")
    device: str = "cpu"
    # Device to store the model on
    storage_device: str = "cpu"
    # Name of the vision encoder model (Set to "lerobot/resnet10" for hil serl resnet10)
    vision_encoder_name: str | None = None
    # Whether to freeze the vision encoder during training
    freeze_vision_encoder: bool = True
    # Hidden dimension size for the image encoder
    image_encoder_hidden_dim: int = 32
    # Whether to use a shared encoder for actor and critic
    shared_encoder: bool = True
    # Number of discrete actions, eg for gripper actions. The discrete
    # dimension is passed through from the base action and never modified by
    # the residual.
    num_discrete_actions: int | None = None
    # Dimension of the image embedding pooling
    image_embedding_pooling_dim: int = 8

    # Encoder architecture
    # Hidden dimension size for the state encoder
    state_encoder_hidden_dim: int = 256
    # Dimension of the latent space
    latent_dim: int = 256

    # Residual actor specifics
    # Residual mode: predict a correction on top of ``observation.base_action``
    # instead of a full action.
    residual: bool = True
    # Path (or Hub ID) of the frozen base policy (e.g. an ACT checkpoint) used
    # by the actor's environment wrapper to produce ``observation.base_action``.
    # Required for rollouts in residual mode; the base weights are never
    # trained and are not part of the algorithm checkpoints.
    base_policy_path: str | None = None
    # Type of the frozen base policy (registry name accepted by
    # ``lerobot.policies.factory.get_policy_class``), e.g. "act" or "diffusion".
    # Any BC policy with ``select_action``/``reset`` and pre/post processors
    # works — the residual method itself is agnostic to the base architecture.
    base_policy_type: str = "act"
    # Bound on the residual magnitude: residual = action_scale * tanh(x)
    action_scale: float = 0.2
    # Init scale for the actor final layer. 0.0 makes the residual start at
    # zero (initial policy == base policy).
    actor_last_layer_init_scale: float | None = 0.0
    # Stddev of the Gaussian exploration noise added to the residual during
    # rollouts (0 disables exploration noise).
    exploration_stddev: float = 0.025
    # Hard clip applied to the exploration noise sample.
    exploration_noise_clip: float = 0.3

    # Online training (TODO(Khalil): relocate to TrainRLServerPipelineConfig)
    # Number of steps for online training
    online_steps: int = 1000000
    # Capacity of the online replay buffer
    online_buffer_capacity: int = 100000
    # Capacity of the offline replay buffer
    offline_buffer_capacity: int = 100000
    # Whether to use asynchronous prefetching for the buffers
    async_prefetch: bool = False
    # Number of steps before learning starts
    online_step_before_learning: int = 100

    # Actor-learner transport (TODO(Khalil): relocate to TrainRLServerPipelineConfig).
    # Configuration for actor-learner architecture
    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)
    # Configuration for concurrency settings (you can use threads or processes for the actor and learner)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    # Network architecture
    # Configuration for the actor network architecture
    actor_network_kwargs: ActorNetworkConfig = field(default_factory=ActorNetworkConfig)

    def __post_init__(self):
        super().__post_init__()

    def get_optimizer_preset(self) -> MultiAdamConfig:
        # Default learning rate used to satisfy the abstract ``get_optimizer_preset()``
        # contract from ``PreTrainedConfig``. The actual optimizers used during RL
        # training are built by ``TD3Algorithm.make_optimizers_and_scheduler()`` from
        # ``TD3AlgorithmConfig.{actor_lr,critic_lr}`` and fully bypass this preset.
        default_lr = 3e-4
        return MultiAdamConfig(
            weight_decay=0.0,
            optimizer_groups={
                "actor": {"lr": default_lr},
                "critic": {"lr": default_lr},
            },
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        has_image = any(is_image_feature(key) for key in self.input_features)
        has_state = OBS_STATE in self.input_features

        if not (has_state or has_image):
            raise ValueError(
                "You must provide either 'observation.state' or an image observation (key starting with 'observation.image') in the input features"
            )

        if ACTION not in self.output_features:
            raise ValueError("You must provide 'action' in the output features")

        if self.residual and OBS_BASE_ACTION not in self.input_features:
            raise ValueError(
                f"TD3ActorConfig(residual=True) requires '{OBS_BASE_ACTION}' in the input "
                "features. The base-policy environment wrapper injects this key during "
                "rollouts, and the offline-buffer labeling pass adds it to demonstrations."
            )

    @property
    def image_features(self) -> list[str]:
        return [key for key in self.input_features if is_image_feature(key)]

    @property
    def observation_delta_indices(self) -> list:
        return None

    @property
    def action_delta_indices(self) -> list:
        return None  # TD3 predicts one action at a time

    @property
    def reward_delta_indices(self) -> None:
        return None
