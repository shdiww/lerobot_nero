"""
Nero — LeRobot Robot 子类, 封装 pyAgxArm SDK 控制 Agilex Nero 7-DOF 机械臂.

pyAgxArm SDK 核心 API 对照:
    AgxArmFactory.create_arm(config)    → 根据配置创建驱动实例 ( NeroDriverDefault / V111 / V112 )
    arm.connect()                       → 建立 CAN 通信连接
    arm.enable(joint_index=255)         → 使能所有关节电机 (255 = 全部)
    arm.disconnect()                    → 断开 CAN 通信
    arm.is_connected()                  → 查询通信是否建立
    arm.get_joints_enable_status_list() → 查询各关节使能状态
    arm.get_joint_angles()              → 读取当前 7 个关节角 (弧度), 返回 .msg = list[float]
    arm.set_speed_percent(p)            → 设置运动速度百分比 (1~100)
    arm.set_motion_mode(mode)           → 设置运动模式: MOTION_MODE.J (插补) / .JS (伺服)
    arm.move_j(joints)                  → 关节插补运动 (先到位再返回, 适合安全场景)
    arm.move_js(joints)                 → 关节伺服运动 (立即下发, 适合实时遥操作)
    arm.set_auto_set_motion_mode_enabled(False) → 禁止 SDK 自动切换运动模式
    arm.init_effector(type)             → 初始化末端执行器 (夹爪)
    effector.get_gripper_status()       → 读取夹爪状态 (.msg.mode, .msg.value)
    effector.move_gripper_m(value, force) → 控制夹爪开合 (value=米制宽度, force=力度)
"""

import logging
import time
from functools import cached_property

from lerobot.cameras import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

from ..robot import Robot
from .config_nero import (
    NERO_GRIPPER_MAX_WIDTH_M,
    NERO_JOINT_LIMITS_RAD,
    NERO_JOINT_NAMES,
    NeroRobotConfig,
)

logger = logging.getLogger(__name__)


def _clamp_joints(joints: list[float], limits: dict[str, list[float]]) -> list[float]:
    clamped = []
    for _i, (name, val) in enumerate(zip(NERO_JOINT_NAMES, joints, strict=True)):
        lo, hi = limits[name]
        clamped.append(max(lo, min(hi, val)))
    return clamped


class Nero(Robot):
    config_class = NeroRobotConfig
    name = "nero"

    def __init__(self, config: NeroRobotConfig):
        super().__init__(config)
        self.config = config
        self.arm = None
        self.effector = None
        self.cameras = make_cameras_from_configs(config.cameras)
        # 根据固件版本 / CAN 接口 / 通道创建 SDK 驱动配置对象,
        # 后续通过 AgxArmFactory.create_arm(config) 生成对应驱动实例.
        self._sdk_config = create_agx_arm_config(
            robot=ArmModel.NERO,
            firmeware_version={"default": NeroFW.DEFAULT, "v111": NeroFW.V111, "v112": NeroFW.V112}.get(
                config.firmware_version, config.firmware_version
            ),
            interface=config.can_interface,
            channel=config.can_channel,
        )

    @property
    def _joints_ft(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in NERO_JOINT_NAMES}

    @property
    def _gripper_ft(self) -> dict[str, type]:
        if self.config.use_gripper:
            return {"gripper.pos": float}
        return {}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._joints_ft, **self._gripper_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {**self._joints_ft, **self._gripper_ft}

    @property
    def is_connected(self) -> bool:
        """检查通信与使能状态.

        SDK 调用链:
            arm.is_connected()                → CAN 通信是否建立
            arm.get_joints_enable_status_list() → 各关节电机是否已使能
            effector.get_gripper_status()      → 夹爪是否已初始化 (如配置)
            cam.is_connected                  → 相机是否已连接
        全部为 True 才认为机器人可用.
        """
        if self.arm is None or not self.arm.is_connected():
            return False
        status_list = self.arm.get_joints_enable_status_list()
        if status_list is None or not all(status_list):
            return False
        if self.config.use_gripper and self.effector is not None:
            gs = self.effector.get_gripper_status()
            if gs is None:
                return False
        return all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """建立 CAN 通信, 使能关节电机, 初始化夹爪 (如配置).

        SDK 调用链:
            1. AgxArmFactory.create_arm(sdk_config) → 创建驱动实例 (self.arm)
            2. arm.init_effector(EFFECTOR.AGX_GRIPPER) → 初始化夹爪 (self.effector)
            3. arm.connect()  → 建立 CAN 总线通信
            4. arm.enable(255) → 使能全部 7 个关节电机 (最多重试 500 次 × 10ms = 5s)
            5. configure()    → 禁止 SDK 自动切换运动模式
            6. cam.connect()  → 逐个连接相机
        """
        try:
            self.arm = AgxArmFactory.create_arm(self._sdk_config)

            if self.config.use_gripper:
                self.effector = self.arm.init_effector(
                    self.arm.OPTIONS.EFFECTOR.AGX_GRIPPER
                )

            self.arm.connect()

            for _ in range(500):
                if self.arm.enable():
                    break
                time.sleep(0.01)
            else:
                logger.warning("Nero enable timed out after 5s")

            self.configure()

            for cam in self.cameras.values():
                cam.connect()

            logger.info(f"{self} connected.")
        except Exception:
            self.arm = None
            self.effector = None
            raise

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def move_to_home(self) -> None:
        """用 move_j 模式运动到 home 位置并闭合夹爪.

        SDK 调用链:
            1. arm.set_speed_percent(10)              → 设速度 10% (安全速度)
            2. arm.set_motion_mode(MOTION_MODE.J)     → 切换到关节插补模式
            3. arm.move_j(home_joint_angles)          → 运动到 home 关节角
            4. effector.move_gripper_m(0.0, force)    → 闭合夹爪
        适用于: episode 间复位、测试验证、手动回零.
        """
        if self.arm is None:
            return
        self.arm.set_speed_percent(40)
        self.arm.set_motion_mode(self.arm.OPTIONS.MOTION_MODE.J)
        self.arm.move_j(self.config.home_joint_angles)
        if self.config.use_gripper and self.effector is not None:
            self.effector.move_gripper_m(value=0.0, force=self.config.gripper_force)

    def configure(self) -> None:
        """允许 SDK 自动切换运动模式.

        SDK 调用:
            arm.set_auto_set_motion_mode_enabled(True)
            启用后 SDK 在调用 move_j/move_js 时会自动切换对应运动模式,
            无需手动调用 set_motion_mode().
        """
        if self.arm is None:
            return
        self.arm.set_auto_set_motion_mode_enabled(True)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """读取当前关节角 / 夹爪状态 / 相机图像.

        SDK 调用链:
            1. arm.get_joint_angles() → 返回 7 个关节当前弧度 (.msg = list[float])
            2. effector.get_gripper_status() → 返回夹爪状态
               .msg.mode == "width" 时 .msg.value 为米制开合宽度 (米)
            3. cam.read_latest() → 读取相机最新帧
        """
        obs_dict: RobotObservation = {}

        ja = self.arm.get_joint_angles()
        if ja is not None:
            for name, val in zip(NERO_JOINT_NAMES, ja.msg, strict=True):
                obs_dict[f"{name}.pos"] = val

        if self.config.use_gripper and self.effector is not None:
            gs = self.effector.get_gripper_status()
            if gs is not None and gs.msg.mode == "width":
                obs_dict["gripper.pos"] = gs.msg.value
            else:
                obs_dict["gripper.pos"] = 0.0

        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.read_latest()

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """下发目标关节角 + 夹爪开合到机械臂.

        处理流程:
            1. 从 action dict 提取 7 个关节目标角
            2. _clamp_joints: 将目标角限制在 NERO_JOINT_LIMITS_RAD 内
            3. _clip_relative: 如配置了 max_relative_target, 限制单步最大变化量
            4. 根据运动模式下发:
               move_j 模式 (更安全, 插补到位):
                   arm.set_speed_percent(10)              → 先设速度
                   arm.set_motion_mode(MOTION_MODE.J)     → 再设模式
                   arm.move_j(goal_joints)                → 最后下发
               move_js 模式 (更快响应, 伺服):
                   arm.set_motion_mode(MOTION_MODE.JS)    → 设模式
                   arm.move_js(goal_joints)               → 下发
            5. 夹爪控制:
               effector.move_gripper_m(value, force)      → 米制宽度 + 力度

        SDK 调用:
            arm.get_joint_angles()   → 读取当前角 (用于 relative clip)
            arm.set_speed_percent()  → 设置速度百分比
            arm.set_motion_mode()    → 切换 J / JS 模式
            arm.move_j() / move_js() → 下发关节目标
            effector.move_gripper_m() → 夹爪控制
        """
        goal_joints = []
        for name in NERO_JOINT_NAMES:
            key = f"{name}.pos"
            if key in action:
                goal_joints.append(action[key])

        if len(goal_joints) == len(NERO_JOINT_NAMES):
            goal_joints = _clamp_joints(goal_joints, NERO_JOINT_LIMITS_RAD)

            if self.config.max_relative_target is not None:
                current_ja = self.arm.get_joint_angles()
                if current_ja is not None:
                    goal_joints = self._clip_relative(
                        goal_joints, current_ja.msg, self.config.max_relative_target
                    )

            if self.config.motion_mode == "j":
                self.arm.set_speed_percent(30)
                # self.arm.set_motion_mode(self.arm.OPTIONS.MOTION_MODE.J)
                self.arm.move_j(goal_joints)
            else:
                # self.arm.set_motion_mode(self.arm.OPTIONS.MOTION_MODE.JS)
                self.arm.move_js(goal_joints)

        if self.config.use_gripper and self.effector is not None and "gripper.pos" in action:
            gripper_val = float(action["gripper.pos"])
            gripper_val = max(0.0, min(NERO_GRIPPER_MAX_WIDTH_M, gripper_val))
            self.effector.move_gripper_m(
                value=gripper_val, force=self.config.gripper_force
            )

        sent_action = {}
        for i, name in enumerate(NERO_JOINT_NAMES):
            if f"{name}.pos" in action:
                sent_action[f"{name}.pos"] = goal_joints[i] if goal_joints else action[f"{name}.pos"]
        if self.config.use_gripper and "gripper.pos" in action:
            sent_action["gripper.pos"] = action["gripper.pos"]

        return sent_action

    @staticmethod
    def _clip_relative(
        goal: list[float],
        current: list[float],
        max_rel: float | dict[str, float],
    ) -> list[float]:
        clamped = []
        for i, name in enumerate(NERO_JOINT_NAMES):
            cap = max_rel if isinstance(max_rel, (int, float)) else max_rel.get(name, 0.5)
            delta = goal[i] - current[i]
            delta = max(-cap, min(cap, delta))
            clamped.append(current[i] + delta)
        return clamped

    @check_if_not_connected
    def disconnect(self):
        """断开 CAN 通信, 释放驱动和夹爪实例.

        SDK 调用:
            arm.disconnect() → 断开 CAN 总线通信
            不调用 arm.disable() 或 arm.electronic_emergency_stop(),
            关节使能/失能由用户通过 Nero 上位机自行控制.
        """
        self.arm.disconnect()
        self.arm = None
        self.effector = None

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
