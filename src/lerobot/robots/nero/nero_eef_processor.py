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

"""RL env processor steps for Nero's absolute-EEF-pose action space.

These steps let a policy live in the 10-dim EEF space (xyz + 6D rotation +
gripper, the convention of the EEF-pose recording / deployment scripts) while
the robot itself is driven in joint space:

- ``NeroEefObservationProcessorStep`` (observation side): replaces the joint
  ``agent_pos`` vector with the FK pose, before
  ``VanillaObservationProcessorStep`` maps it to ``observation.state``.
- ``NeroEefActionToJointsProcessorStep`` (action side): converts a 10-dim
  absolute EEF action into a joint-target vector via seeded, per-step clamped
  IK, holding the last command on IK failure — the same behavior as the
  deployment script.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStep, ProcessorStepRegistry, TransitionKey

from .eef_utils import EEF_POSE_KEYS, eef_dict_to_joints, joints_to_eef_dict

logger = logging.getLogger(__name__)


@ProcessorStepRegistry.register("nero_eef_observation")
@dataclass
class NeroEefObservationProcessorStep(ProcessorStep):
    """Replace the joint ``agent_pos`` with the 10-dim EEF state via FK.

    Runs BEFORE ``VanillaObservationProcessorStep`` (which maps ``agent_pos``
    to ``observation.state``). The raw ``{joint}.pos`` keys are left in the
    observation dict; downstream consumers select what they need.
    """

    adapter: Any
    arm_joint_names: list[str]
    gripper_key: str = "gripper.pos"

    def __call__(self, transition):
        obs = transition.get(TransitionKey.OBSERVATION)
        if obs is None or "agent_pos" not in obs:
            return transition

        joints = [float(obs[f"{name}.pos"]) for name in self.arm_joint_names]
        eef = joints_to_eef_dict(self.adapter, joints)
        gripper = float(obs.get(self.gripper_key, 0.0))

        new_transition = dict(transition)
        new_obs = dict(obs)
        new_obs["agent_pos"] = np.array(
            [eef[key] for key in EEF_POSE_KEYS] + [gripper], dtype=np.float32
        )
        new_transition[TransitionKey.OBSERVATION] = new_obs
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # The joint entries are replaced by the EEF pose dims (gripper passes through)
        for name in self.arm_joint_names:
            features[PipelineFeatureType.OBSERVATION].pop(f"{name}.pos", None)
        for key in EEF_POSE_KEYS:
            features[PipelineFeatureType.OBSERVATION][key] = PolicyFeature(
                type=FeatureType.STATE, shape=(1,)
            )
        return features


@ProcessorStepRegistry.register("nero_eef_action_to_joints")
@dataclass
class NeroEefActionToJointsProcessorStep(ProcessorStep):
    """Convert a 10-dim absolute EEF action into a joint-target vector via IK.

    Behavior mirrors the deployment script: IK is seeded with the currently
    measured arm joints (read from the transition's observation), every
    commanded joint change is clamped to ``max_joint_step`` radians for
    smoothness/safety, and on IK failure the last commanded joints are held.
    The gripper component passes through unchanged. The output vector is
    ordered by ``motor_names`` as expected by ``RobotEnv.step``.
    """

    adapter: Any
    arm_joint_names: list[str]
    motor_names: list[str]
    gripper_key: str = "gripper.pos"
    max_joint_step: float = 0.15
    _last_joints: np.ndarray | None = field(default=None, init=False, repr=False)

    def __call__(self, transition):
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            raise ValueError("NeroEefActionToJoints requires an action in the transition")
        action_flat = np.asarray(action, dtype=np.float64).reshape(-1)
        if action_flat.shape[0] != len(EEF_POSE_KEYS) + 1:
            raise ValueError(
                f"Expected a {len(EEF_POSE_KEYS) + 1}-dim EEF action, got shape {action_flat.shape}"
            )

        obs = transition.get(TransitionKey.OBSERVATION) or {}
        try:
            seed = np.array(
                [float(obs[f"{name}.pos"]) for name in self.arm_joint_names], dtype=np.float64
            )
        except KeyError:
            seed = None
        if seed is None and self._last_joints is None:
            raise ValueError(
                "NeroEefActionToJoints requires the measured joint positions in the "
                "transition observation for the IK seed (or a previously held command)"
            )
        if seed is None:
            seed = self._last_joints

        eef = {key: float(value) for key, value in zip(EEF_POSE_KEYS, action_flat[:-1], strict=True)}
        joints = eef_dict_to_joints(self.adapter, eef, seed, self.max_joint_step)
        if joints is None:
            # Hold the last commanded joints (or the seed on the very first step)
            logger.warning("Nero EEF IK failed; holding the last commanded joints")
            joints = (
                self._last_joints.copy()
                if self._last_joints is not None
                else np.asarray(seed, dtype=np.float64).copy()
            )
        self._last_joints = joints.copy()

        joint_targets = {name: float(joints[i]) for i, name in enumerate(self.arm_joint_names)}
        joint_targets[self.gripper_key.removesuffix(".pos")] = float(action_flat[-1])

        processed = np.array(
            [joint_targets[name.removesuffix(".pos")] for name in self.motor_names], dtype=np.float32
        )

        new_transition = dict(transition)
        new_transition[TransitionKey.ACTION] = processed
        return new_transition

    def reset(self) -> None:
        self._last_joints = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # The env-facing action becomes a joint-target vector ordered by motor_names
        features[PipelineFeatureType.ACTION] = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(len(self.motor_names),))
        }
        return features
