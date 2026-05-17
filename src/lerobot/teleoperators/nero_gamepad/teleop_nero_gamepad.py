import logging
import os
from typing import Any

import numpy as np

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_not_connected

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_nero_gamepad import NeroGamepadConfig
from .nero_gamepad_utils import NeroGamepadController

logger = logging.getLogger(__name__)


def _try_create_tracik_solver(
    urdf_path: str,
    base_link_name: str = "base_link",
    target_link_name: str = "gripper_base",
    timeout: float = 0.005,
    epsilon: float = 1e-5,
    solver_type: str = "Speed",
):
    try:
        from trac_ik import TracIK
    except ImportError:
        logger.warning("trac_ik not installed — IK unavailable")
        return None

    if not os.path.exists(urdf_path):
        logger.warning(f"URDF not found: {urdf_path}")
        return None

    try:
        solver = TracIK(
            base_link_name=base_link_name,
            tip_link_name=target_link_name,
            urdf_path=urdf_path,
            timeout=timeout,
            epsilon=epsilon,
            solver_type=solver_type,
        )
        logger.info(f"TracIK created: {base_link_name} -> {target_link_name}, joints={len(solver.joint_limits[0])}")
        return solver
    except Exception as e:
        logger.warning(f"TracIK init failed: {e}")
        return None


class TracIKAdapter:
    def __init__(self, solver):
        self.solver = solver
        self._seed = np.zeros(len(solver.joint_limits[0]))

    def fk(self, joint_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos, rot = self.solver.fk(np.asarray(joint_angles, dtype=np.float64))
        return np.asarray(pos, dtype=np.float64), np.asarray(rot, dtype=np.float64)

    def ik(self, target_pos: np.ndarray, target_rot: np.ndarray, seed: np.ndarray | None = None) -> np.ndarray | None:
        if seed is None:
            seed = self._seed
        result = self.solver.ik(
            np.asarray(target_pos, dtype=np.float64),
            np.asarray(target_rot, dtype=np.float64),
            np.asarray(seed, dtype=np.float64),
        )
        if result is not None:
            self._seed = np.asarray(result, dtype=np.float64)
        return result

    def set_seed(self, joint_angles: list[float] | np.ndarray) -> None:
        self._seed = np.asarray(joint_angles, dtype=np.float64)


class NeroGamepad(Teleoperator):
    config_class = NeroGamepadConfig
    name = "nero_gamepad"

    def __init__(self, config: NeroGamepadConfig):
        super().__init__(config)
        self.config = config
        self.controller = None

        self.ik_solver = None
        self._ik_initialized = False
        self._joint_angles = np.zeros(7)
        self._ee_pos = np.zeros(3)
        self._ee_rot = np.eye(3)
        self._ema_joint_angles = None
        self.gripper_state = 50.0

    @property
    def action_features(self) -> dict[str, type]:
        features = {
            "delta_x": float,
            "delta_y": float,
            "delta_z": float,
            "delta_wx": float,
            "delta_wy": float,
            "delta_wz": float,
        }
        if self.config.use_gripper:
            features["delta_gripper"] = float
        return features

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.controller is not None and self.controller.joystick_connected

    def connect(self, calibrate: bool = True) -> None:
        self.controller = NeroGamepadController(self.config)
        self.controller.start()

        if self.config.urdf_path and os.path.exists(self.config.urdf_path):
            raw_solver = _try_create_tracik_solver(
                self.config.urdf_path,
                self.config.base_link_name,
                self.config.target_link_name,
            )
            if raw_solver is not None:
                self.ik_solver = TracIKAdapter(raw_solver)
                self.set_joint_angles(self.config.home_joint_angles)
                self.ik_solver.set_seed(self.config.home_joint_angles)
                logger.info("IK initialized")
            else:
                logger.warning("IK init failed — teleop will output deltas only")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        self.controller.update()

        delta_x, delta_y, delta_z, delta_wx, delta_wy, delta_wz = self.controller.get_deltas()

        action: RobotAction = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
            "delta_wx": delta_wx,
            "delta_wy": delta_wy,
            "delta_wz": delta_wz,
        }

        if self.config.use_gripper:
            action["delta_gripper"] = self.controller.gripper_command()

        return action

    def get_teleop_events(self) -> dict[str, Any]:
        if self.controller is None:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        is_intervention = True

        episode_end_status = self.controller.get_episode_end_status()
        terminate_episode = episode_end_status in [
            TeleopEvents.RERECORD_EPISODE,
            TeleopEvents.FAILURE,
        ]
        success = episode_end_status == TeleopEvents.SUCCESS
        rerecord_episode = episode_end_status == TeleopEvents.RERECORD_EPISODE

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }

    def is_home_requested(self) -> bool:
        if self.controller is None:
            return False
        return self.controller.is_home_requested()

    def set_joint_angles(self, joint_angles: list[float] | np.ndarray) -> None:
        self._joint_angles = np.asarray(joint_angles, dtype=np.float64).copy()
        self._ema_joint_angles = self._joint_angles.copy()
        if self.ik_solver is not None:
            self._ee_pos, self._ee_rot = self.ik_solver.fk(self._joint_angles)
            self._ik_initialized = True

    def get_joint_angles(self) -> list[float]:
        return self._joint_angles.tolist()

    def solve_ik_from_deltas(
        self,
        delta_x: float,
        delta_y: float,
        delta_z: float,
        delta_wx: float,
        delta_wy: float,
        delta_wz: float,
    ) -> bool:
        if self.ik_solver is None or not self._ik_initialized:
            return False

        try:
            from scipy.spatial.transform import Rotation
        except ImportError:
            logger.error("scipy not installed — cannot compute rotation deltas for IK")
            return False

        new_pos = self._ee_pos.copy()
        new_pos[0] += delta_x
        new_pos[1] += delta_y
        new_pos[2] += delta_z

        r = Rotation.from_matrix(self._ee_rot)
        euler = r.as_euler("xyz")
        euler[0] += delta_wx
        euler[1] += delta_wy
        euler[2] += delta_wz
        new_rot = Rotation.from_euler("xyz", euler).as_matrix()

        seed = self._joint_angles.copy()
        result = self.ik_solver.ik(new_pos, new_rot, seed)

        if result is None:
            logger.debug("IK failed — no solution found")
            return False

        result = np.asarray(result, dtype=np.float64)

        delta_joints = result - self._joint_angles
        if np.any(np.abs(delta_joints) > self.config.reject_threshold):
            logger.debug("IK rejected — joint delta too large")
            return False

        if self.config.max_joint_step > 0:
            delta_joints = np.clip(delta_joints, -self.config.max_joint_step, self.config.max_joint_step)
            result = self._joint_angles + delta_joints

        alpha = self.config.ema_alpha
        if self._ema_joint_angles is not None:
            self._ema_joint_angles = alpha * result + (1 - alpha) * self._ema_joint_angles
        else:
            self._ema_joint_angles = result.copy()

        self._joint_angles = self._ema_joint_angles.copy()
        self._ee_pos, self._ee_rot = self.ik_solver.fk(self._joint_angles)
        return True

    def update_gripper_state(self, delta_gripper: float) -> None:
        self.gripper_state = max(0.0, min(100.0, self.gripper_state + delta_gripper))

    def send_feedback(self, feedback: dict) -> None:
        pass

    def disconnect(self) -> None:
        if self.controller is not None:
            self.controller.stop()
            self.controller = None
