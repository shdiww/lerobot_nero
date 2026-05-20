import logging
import time

from lerobot.utils.import_utils import _pygame_available, require_package

if _pygame_available:
    import pygame
else:
    pygame = None

from ..utils import TeleopEvents

logger = logging.getLogger(__name__)


class NeroGamepadController:
    def __init__(self, config):
        require_package("pygame", extra="gamepad")
        self.config = config
        self.joystick = None
        self.joystick_connected = False
        self.running = True

        self.episode_end_status = None
        self.open_gripper_command = False
        self.close_gripper_command = False
        self.home_requested = False
        self.start_pressed = False
        self.back_pressed = False

        self.button_map = {
            "a": 0, "b": 1, "x": 2, "y": 3,
            "lb": 4, "rb": 5, "back": 6, "start": 7,
            "home": 8, "l3": 9, "r3": 10,
        }
        self.axis_map = {
            "left_x": 0, "left_y": 1,
            "right_x": 3, "right_y": 4,
            "left_trigger": 2, "right_trigger": 5,
        }
        self.hat_map = {"dpad": 0}

        self._prev_buttons = dict.fromkeys(self.button_map, False)

    def clear_all_events(self):
        self.episode_end_status = None
        self.start_pressed = False
        self.back_pressed = False
        self.home_requested = False
        self.open_gripper_command = False
        self.close_gripper_command = False

    def start(self):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            logger.warning("No gamepad detected. Waiting for connection...")

        self._poll_for_joystick(timeout=10.0)

        logger.info("Nero Gamepad controls:")
        logger.info("  Left stick: End-effector X/Y translation")
        logger.info("  Right stick: Z translation / Z-axis rotation")
        logger.info("  D-pad: X/Y rotation")
        logger.info("  A: Close gripper")
        logger.info("  B: Open gripper")
        logger.info("  Y: Go Home")
        logger.info("  Start: Begin recording")
        logger.info("  Back: End recording / E-STOP")
        logger.info("  Home: Connect robot (script-level)")

    def _poll_for_joystick(self, timeout: float = 10.0):
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            for event in pygame.event.get():
                if event.type == pygame.JOYDEVICEADDED:
                    try:
                        self.joystick = pygame.joystick.Joystick(event.device_index)
                        self.joystick.init()
                        self.joystick_connected = True
                        logger.info(f"Gamepad connected: {self.joystick.get_name()}")
                        return
                    except Exception:
                        self.joystick = None
                        self.joystick_connected = False
            time.sleep(0.1)
        logger.warning("No gamepad found within timeout")

    def stop(self):
        if pygame and pygame.joystick.get_init():
            if self.joystick:
                self.joystick.quit()
            pygame.joystick.quit()
        if pygame:
            pygame.quit()
        self.joystick = None
        self.joystick_connected = False

    def update(self):
        if not pygame:
            return

        for event in pygame.event.get():
            if event.type == pygame.JOYDEVICEADDED:
                try:
                    self.joystick = pygame.joystick.Joystick(event.device_index)
                    self.joystick.init()
                    self.joystick_connected = True
                except Exception:
                    self.joystick = None
                    self.joystick_connected = False
            elif event.type == pygame.JOYDEVICEREMOVED:
                self.joystick = None
                self.joystick_connected = False

        if not self.joystick_connected or self.joystick is None:
            return

        try:
            num_btns = self.joystick.get_numbuttons()
            for name, btn_id in self.button_map.items():
                if btn_id >= num_btns:
                    if name in ("start", "back", "home"):
                        print(f"\n[GAMEPAD] Button '{name}' (id={btn_id}) >= numbuttons={num_btns}, SKIPPED!")
                    continue
                current = bool(self.joystick.get_button(btn_id))
                prev = self._prev_buttons.get(name, False)
                if current and not prev:
                    print(f"\n[GAMEPAD] Button pressed: {name} (id={btn_id})")
                    if name == "y":
                        self.episode_end_status = TeleopEvents.SUCCESS
                    elif name == "back":
                        self.back_pressed = True
                    elif name == "start":
                        self.start_pressed = True
                    elif name == "home":
                        self.home_requested = True
                self._prev_buttons[name] = current

            self.close_gripper_command = bool(self.joystick.get_button(self.button_map["a"]))
            self.open_gripper_command = bool(self.joystick.get_button(self.button_map["b"]))

        except pygame.error:
            self.joystick_connected = False
            self.joystick = None

    def get_deltas(self):
        if not self.joystick_connected or self.joystick is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        try:
            left_x = self._apply_deadzone(self._get_axis_value("left_x"))
            left_y = self._apply_deadzone(self._get_axis_value("left_y"))
            right_x = self._apply_deadzone(self._get_axis_value("right_x"))
            right_y = self._apply_deadzone(self._get_axis_value("right_y"))
            hat = self._get_hat_value("dpad")

            cfg = self.config
            sf = cfg.speed_factor
            delta_x = left_y * cfg.x_step_size * sf
            delta_y = left_x * cfg.y_step_size * sf
            delta_z = -right_y * cfg.z_step_size * sf
            delta_wx = hat[0] * cfg.wx_step_size * sf
            delta_wy = -hat[1] * cfg.wy_step_size * sf
            delta_wz = right_x * cfg.wz_step_size * sf

            return delta_x, delta_y, delta_z, delta_wx, delta_wy, delta_wz

        except pygame.error:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def gripper_command(self):
        delta = 0.0
        if self.close_gripper_command:
            delta -= self.config.gripper_step * self.config.speed_factor
        if self.open_gripper_command:
            delta += self.config.gripper_step * self.config.speed_factor
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

    def _apply_deadzone(self, value: float) -> float:
        return value if abs(value) > self.config.deadzone else 0.0

    def _get_axis_value(self, axis_name: str) -> float:
        if axis_name not in self.axis_map:
            return 0.0
        axis_index = self.axis_map[axis_name]
        if axis_index >= self.joystick.get_numaxes():
            return 0.0
        value = self.joystick.get_axis(axis_index)
        if axis_name in ("left_trigger", "right_trigger"):
            return (value + 1) / 2
        return value

    def _get_hat_value(self, hat_name: str) -> tuple[int, int]:
        if hat_name not in self.hat_map:
            return (0, 0)
        hat_index = self.hat_map[hat_name]
        if hat_index >= self.joystick.get_numhats():
            return (0, 0)
        return self.joystick.get_hat(hat_index)
