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

"""Nero EEF pose utilities: joint angles <-> end-effector pose.

Package-side copy of the math in ``scripts/nero_eef_utils.py`` (outer repo) so
the lerobot RL env can use the exact same conventions as the EEF-pose
recording / deployment scripts. Keep both in sync.

EEF pose convention (rotation in the 6D continuous representation, i.e. the
first two rows of the rotation matrix):
    [x, y, z, r00, r01, r02, r10, r11, r12, gripper]  -- 10 dims
    - position: base_link frame, meters
    - orientation: R[:2, :].flatten() (Zhou et al. 6D rep; Nero's tool
      orientation sits near the +/-pi wrap boundary for every euler order,
      hence not euler/quaternion)
    - gripper: passed through in motor units (same convention as the recorded
      datasets; no conversion here)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

EEF_POSE_KEYS = [
    "eef.x",
    "eef.y",
    "eef.z",
    "eef.r00",
    "eef.r01",
    "eef.r02",
    "eef.r10",
    "eef.r11",
    "eef.r12",
]
EEF_KEYS = [*EEF_POSE_KEYS, "gripper.pos"]


def joints_to_eef_dict(ik_adapter, joints) -> dict[str, float]:
    """FK: 7 joint angles -> {eef.x/y/z, eef.r00..r12} (no gripper)."""
    pos, rot = ik_adapter.fk(np.asarray(joints, dtype=np.float64))
    r6 = np.asarray(rot, dtype=np.float64)[:2, :].flatten()
    return {
        "eef.x": float(pos[0]),
        "eef.y": float(pos[1]),
        "eef.z": float(pos[2]),
        "eef.r00": float(r6[0]),
        "eef.r01": float(r6[1]),
        "eef.r02": float(r6[2]),
        "eef.r10": float(r6[3]),
        "eef.r11": float(r6[4]),
        "eef.r12": float(r6[5]),
    }


def _rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """6D representation -> rotation matrix (Gram-Schmidt, Zhou et al. 2018)."""
    a1 = np.asarray(r6[0:3], dtype=np.float64)
    a2 = np.asarray(r6[3:6], dtype=np.float64)
    b1 = a1 / (np.linalg.norm(a1) + 1e-9)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0)


def eef_dict_to_joints(ik_adapter, eef: dict[str, float], seed_joints, max_joint_step: float = 0.15):
    """IK: {eef.*} -> 7 joint angles. Returns None if unsolved; otherwise the
    solution is clamped to ``max_joint_step`` around the seed for smoothness.

    The eef dict only needs the 9 pose keys (xyz + 6D rotation); the gripper is
    handled by the caller.
    """
    pos = np.array([eef["eef.x"], eef["eef.y"], eef["eef.z"]], dtype=np.float64)
    r6 = np.array([eef[k] for k in EEF_POSE_KEYS[3:]], dtype=np.float64)
    rot = _rot6d_to_matrix(r6)

    seed = np.asarray(seed_joints, dtype=np.float64)
    result = ik_adapter.ik(pos, rot, seed=seed)
    if result is None:
        return None
    result = np.asarray(result, dtype=np.float64)

    if max_joint_step and max_joint_step > 0:
        result = seed + np.clip(result - seed, -max_joint_step, max_joint_step)
    return result


def create_eef_ik_adapter(
    urdf_path: str | None = None,
    base_link_name: str = "base_link",
    target_link_name: str = "gripper_base",
    timeout: float = 0.005,
):
    """Create the TracIK adapter (shared with the nero gamepad teleop); None on failure."""
    from lerobot.teleoperators.nero_gamepad.teleop_nero_gamepad import (
        TracIKAdapter,
        _try_create_tracik_solver,
    )

    if urdf_path is None:
        from lerobot.robots.nero.config_nero import NERO_URDF_PATH

        urdf_path = str(NERO_URDF_PATH)

    solver = _try_create_tracik_solver(
        urdf_path,
        base_link_name=base_link_name,
        target_link_name=target_link_name,
        timeout=timeout,
    )
    if solver is None:
        return None
    return TracIKAdapter(solver)
