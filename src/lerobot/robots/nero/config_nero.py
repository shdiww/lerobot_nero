import json
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig

from ..config import RobotConfig

NERO_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]

NERO_JOINT_LIMITS_RAD = {
    "joint1": [-2.705261, 2.705261],
    "joint2": [-1.745330, 1.745330],
    "joint3": [-2.757621, 2.757621],
    "joint4": [-1.012291, 2.146755],
    "joint5": [-2.757621, 2.757621],
    "joint6": [-0.733039, 0.959932],
    "joint7": [-1.570797, 1.570797],
}

NERO_DIR = Path(__file__).resolve().parent

NERO_URDF_PATH = NERO_DIR / "nero_description" / "urdf" / "nero_with_gripper.urdf"

NERO_MESH_DIR = NERO_DIR / "nero_description" / "meshes"

_NERO_HOME_JOINTS_PATH = NERO_DIR / "home_joints.json"
with open(_NERO_HOME_JOINTS_PATH) as _f:
    NERO_HOME_JOINT_ANGLES = json.load(_f)["home_joints_rad"]

NERO_GRIPPER_MAX_WIDTH_M = 0.07


@dataclass
class NeroConfig:
    can_channel: str = "can0"
    can_interface: str = "socketcan"
    firmware_version: str = "default"
    motion_mode: str = "j"
    disable_torque_on_disconnect: bool = True
    max_relative_target: float | dict[str, float] | None = None
    use_gripper: bool = True
    gripper_force: float = 1.0
    home_joint_angles: list[float] = field(default_factory=lambda: list(NERO_HOME_JOINT_ANGLES))
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


@RobotConfig.register_subclass("nero")
@dataclass
class NeroRobotConfig(RobotConfig, NeroConfig):
    pass
