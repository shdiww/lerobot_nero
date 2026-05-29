import logging
import time

import numpy as np
from scipy.spatial.transform import Rotation

from ..utils import TeleopEvents

logger = logging.getLogger(__name__)

R_HEADSET_TO_WORLD = np.array(
    [
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ],
    dtype=np.float64,
)


def _quat_diff_as_angle_axis(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> np.ndarray:
    r1 = Rotation.from_quat([q1_wxyz[1], q1_wxyz[2], q1_wxyz[3], q1_wxyz[0]])
    r2 = Rotation.from_quat([q2_wxyz[1], q2_wxyz[2], q2_wxyz[3], q2_wxyz[0]])
    delta_r = r2 * r1.inv()
    return delta_r.as_rotvec()


class PicoVRController:
    def __init__(self, config):
        self.config = config
        self.sdk = None
        self.is_connected = False

        self._active = False
        self._ref_controller_xyz = None
        self._ref_controller_quat_wxyz = None
        self._ref_controller_rot_world = None
        self._controller_rot_world = np.eye(3, dtype=np.float64)

        self._delta_pos = np.zeros(3, dtype=np.float64)
        self._delta_rot = np.zeros(3, dtype=np.float64)

        self.close_gripper_command = False
        self.open_gripper_command = False
        self.home_requested = False
        self.start_pressed = False
        self.back_pressed = False
        self.episode_end_status = None

        self._prev_a = False
        self._prev_b = False
        self._prev_x = False
        self._prev_y = False

    def start(self):
        try:
            import xrobotoolkit_sdk as xrt
        except ImportError:
            raise ImportError(
                "xrobotoolkit_sdk not installed. "
                "Run: cd XRoboToolkit-PC-Service-Pybind && bash setup_ubuntu.sh"
            )

        self.sdk = xrt
        try:
            self.sdk.init()
            self.is_connected = True
        except Exception as e:
            logger.warning(f"Pico SDK init failed: {e}")
            self.is_connected = False
            raise

        ctrl = self.config.controller
        logger.info(f"Pico VR controller started ({ctrl})")
        logger.info("  Grip (>0.9): Activate 6D tracking")
        logger.info("  A: Close gripper")
        logger.info("  B: Open gripper")
        logger.info("  X: Go Home")
        logger.info("  Y: End episode (success)")
        logger.info("  Right menu: Start recording")
        logger.info("  Left menu: Back / stop recording")

    def stop(self):
        if self.sdk is not None:
            try:
                self.sdk.close()
            except Exception:
                pass
        self.sdk = None
        self.is_connected = False

    def update(self):
        if not self.is_connected or self.sdk is None:
            return

        self._update_buttons()
        self._update_tracking()

    def _update_buttons(self):
        try:
            a_cur = bool(self.sdk.get_A_button())
            b_cur = bool(self.sdk.get_B_button())
            x_cur = bool(self.sdk.get_X_button())
            y_cur = bool(self.sdk.get_Y_button())
        except Exception:
            return

        if x_cur and not self._prev_x:
            self.home_requested = True
        if y_cur and not self._prev_y:
            self.episode_end_status = TeleopEvents.SUCCESS

        self._prev_a = a_cur
        self._prev_b = b_cur
        self._prev_x = x_cur
        self._prev_y = y_cur

        self.close_gripper_command = a_cur
        self.open_gripper_command = b_cur

        try:
            right_menu = bool(self.sdk.get_right_menu_button())
            left_menu = bool(self.sdk.get_left_menu_button())
        except Exception:
            right_menu = False
            left_menu = False

        if right_menu:
            self.start_pressed = True
        if left_menu:
            self.back_pressed = True

    def _update_tracking(self):
        try:
            if self.config.controller == "right":
                pose = self.sdk.get_right_controller_pose()
                grip_val = self.sdk.get_right_grip()
            else:
                pose = self.sdk.get_left_controller_pose()
                grip_val = self.sdk.get_left_grip()
        except Exception as e:
            logger.debug(f"Failed to read controller: {e}")
            self._delta_pos = np.zeros(3)
            self._delta_rot = np.zeros(3)
            return

        if pose is None or len(pose) != 7:
            self._delta_pos = np.zeros(3)
            self._delta_rot = np.zeros(3)
            return

        controller_xyz = np.array(pose[:3], dtype=np.float64)
        controller_quat_wxyz = np.array([pose[6], pose[3], pose[4], pose[5]], dtype=np.float64)

        if np.linalg.norm(controller_quat_wxyz) < 1e-6:
            self._delta_pos = np.zeros(3)
            self._delta_rot = np.zeros(3)
            return

        controller_xyz_world = R_HEADSET_TO_WORLD @ controller_xyz

        r_ctrl_headset = Rotation.from_quat(
            [controller_quat_wxyz[1], controller_quat_wxyz[2], controller_quat_wxyz[3], controller_quat_wxyz[0]]
        )
        r_ctrl_world = Rotation.from_matrix(R_HEADSET_TO_WORLD) * r_ctrl_headset
        self._controller_rot_world = r_ctrl_world.as_matrix()

        is_gripping = grip_val > self.config.grip_threshold

        if is_gripping:
            if not self._active:
                logger.info("Pico VR tracking ACTIVATED")
                self._active = True
                self._ref_controller_xyz = controller_xyz_world.copy()
                self._ref_controller_quat_wxyz = controller_quat_wxyz.copy()
                self._ref_controller_rot_world = self._controller_rot_world.copy()
                self._delta_pos = np.zeros(3)
                self._delta_rot = np.zeros(3)
            else:
                self._delta_pos = (
                    (controller_xyz_world - self._ref_controller_xyz) * self.config.scale_factor
                )
        else:
            if self._active:
                logger.info("Pico VR tracking DEACTIVATED")
            self._active = False
            self._ref_controller_xyz = None
            self._ref_controller_quat_wxyz = None
            self._ref_controller_rot_world = None
            self._delta_pos = np.zeros(3)
            self._delta_rot = np.zeros(3)

    def get_deltas(self):
        return (
            float(self._delta_pos[0]),
            float(self._delta_pos[1]),
            float(self._delta_pos[2]),
            float(self._delta_rot[0]),
            float(self._delta_rot[1]),
            float(self._delta_rot[2]),
        )

    def get_abs_rot_matrix(self):
        return self._controller_rot_world.copy()

    @property
    def is_active(self):
        return self._active

    def gripper_command(self):
        delta = 0.0
        if self.close_gripper_command:
            delta -= self.config.gripper_step
        if self.open_gripper_command:
            delta += self.config.gripper_step
        if delta != 0.0:
            logger.info(f"gripper_command returning {delta:.2f} (close={self.close_gripper_command} open={self.open_gripper_command})")
        return delta

    def get_episode_end_status(self):
        status = self.episode_end_status
        self.episode_end_status = None
        return status

    def is_home_requested(self):
        requested = self.home_requested
        self.home_requested = False
        return requested

    def is_start_pressed(self):
        pressed = self.start_pressed
        self.start_pressed = False
        return pressed

    def is_back_pressed(self):
        pressed = self.back_pressed
        self.back_pressed = False
        return pressed

    def clear_all_events(self):
        self.episode_end_status = None
        self.start_pressed = False
        self.back_pressed = False
        self.home_requested = False
        self.close_gripper_command = False
        self.open_gripper_command = False
