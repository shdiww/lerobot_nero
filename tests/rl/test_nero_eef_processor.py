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
"""Tests for the Nero EEF processor steps (FK observation, IK action)."""

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lerobot.processor import TransitionKey  # noqa: E402
from lerobot.robots.nero.nero_eef_processor import (  # noqa: E402
    NeroEefActionToJointsProcessorStep,
    NeroEefObservationProcessorStep,
)

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
MOTOR_NAMES = [*ARM_JOINTS, "gripper"]


class FakeAdapter:
    """Deterministic fake kinematics: FK is a linear map, IK returns a
    solution 1 rad away from the seed; an x-position above 100 marks an
    unreachable (failing) target."""

    def fk(self, joints):
        pos = np.asarray(joints[:3], dtype=np.float64)
        return pos, np.eye(3)

    def ik(self, target_pos, target_rot, seed=None):
        if target_pos[0] > 100.0:
            return None
        return np.asarray(seed, dtype=np.float64) + 1.0


def _joint_obs(joints=(0.0,) * 7, gripper=0.5):
    obs = {f"{name}.pos": float(v) for name, v in zip(ARM_JOINTS, joints, strict=True)}
    obs["gripper.pos"] = gripper
    obs["agent_pos"] = np.array([*joints, gripper], dtype=np.float32)
    return obs


def test_observation_step_replaces_agent_pos_with_eef():
    step = NeroEefObservationProcessorStep(adapter=FakeAdapter(), arm_joint_names=ARM_JOINTS)
    transition = {TransitionKey.OBSERVATION: _joint_obs(joints=(1.0, 2.0, 3.0, 0, 0, 0, 0), gripper=0.5)}

    out = step(transition)
    agent_pos = out[TransitionKey.OBSERVATION]["agent_pos"]
    # [x=1, y=2, z=3, identity-6D rotation, gripper passthrough]
    expected = np.array([1.0, 2.0, 3.0, 1, 0, 0, 0, 1, 0, 0.5], dtype=np.float32)
    assert agent_pos.shape == (10,)
    assert np.allclose(agent_pos, expected)
    # Raw joint keys remain available for downstream (e.g. the IK seed)
    assert out[TransitionKey.OBSERVATION]["joint1.pos"] == 1.0


def test_observation_step_passthrough_without_observation():
    step = NeroEefObservationProcessorStep(adapter=FakeAdapter(), arm_joint_names=ARM_JOINTS)
    assert step({}) == {}


def _action_step():
    return NeroEefActionToJointsProcessorStep(
        adapter=FakeAdapter(),
        arm_joint_names=ARM_JOINTS,
        motor_names=MOTOR_NAMES,
        max_joint_step=0.15,
    )


def _eef_action(x=1.0, gripper=0.7):
    # [x, y, z, 6D identity rotation, gripper]
    return torch.tensor([x, 0.0, 0.0, 1, 0, 0, 0, 1, 0, gripper], dtype=torch.float32)


def test_action_step_converts_clamps_and_passes_gripper():
    step = _action_step()
    transition = {
        TransitionKey.ACTION: _eef_action(gripper=0.7).unsqueeze(0),  # batched like policy output
        TransitionKey.OBSERVATION: _joint_obs(joints=(0.0,) * 7),
    }

    out = step(transition)
    processed = out[TransitionKey.ACTION]
    assert processed.shape == (8,)
    # IK solution is +1 rad from the seed; the 0.15 rad clamp caps it
    assert np.allclose(processed[:7], np.full(7, 0.15))
    assert processed[-1] == pytest.approx(0.7)


def test_action_step_holds_last_command_on_ik_failure():
    step = _action_step()
    # Unreachable target (x > 100) with no previous command: hold the seed
    transition = {
        TransitionKey.ACTION: _eef_action(x=200.0, gripper=0.1),
        TransitionKey.OBSERVATION: _joint_obs(joints=(0.0,) * 7),
    }
    out = step(transition)
    assert np.allclose(out[TransitionKey.ACTION][:7], 0.0)
    assert out[TransitionKey.ACTION][-1] == pytest.approx(0.1)

    # A solvable command sets the held joints to 0.15 (clamped)
    step({TransitionKey.ACTION: _eef_action(), TransitionKey.OBSERVATION: _joint_obs()})

    # Another failure without an observation holds the last command
    out = step({TransitionKey.ACTION: _eef_action(x=200.0, gripper=0.2)})
    assert np.allclose(out[TransitionKey.ACTION][:7], 0.15)
    assert out[TransitionKey.ACTION][-1] == pytest.approx(0.2)


def test_action_step_reset_clears_held_command():
    step = _action_step()
    step({TransitionKey.ACTION: _eef_action(), TransitionKey.OBSERVATION: _joint_obs()})
    step.reset()
    with pytest.raises(ValueError, match="IK seed"):
        step({TransitionKey.ACTION: _eef_action(x=200.0)})


def test_action_step_rejects_wrong_dimension():
    step = _action_step()
    with pytest.raises(ValueError, match="10-dim EEF action"):
        step({TransitionKey.ACTION: torch.zeros(8), TransitionKey.OBSERVATION: _joint_obs()})
