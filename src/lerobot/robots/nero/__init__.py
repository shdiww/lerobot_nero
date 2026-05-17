# Nero 机器人模块 — LeRobot 适配层
#
# 导出:
#   Nero            — Robot 子类，封装 pyAgxArm SDK，实现 LeRobot Robot 接口
#   NeroConfig      — 用户面向的配置 dataclass（CAN、运动模式、夹爪等）
#   NeroRobotConfig — 合并 RobotConfig + NeroConfig，注册为 "nero" 子类型
#                     使 CLI 可用 --robot.type=nero
from .config_nero import NeroConfig, NeroRobotConfig
from .nero import Nero

__all__ = [
    "Nero",
    "NeroConfig",
    "NeroRobotConfig",
]
