import json
from dataclasses import dataclass, field
from pathlib import Path

from ..config import TeleoperatorConfig

_NERO_DIR = Path(__file__).resolve().parents[2] / "robots" / "nero"
_DEFAULT_URDF = str(_NERO_DIR / "nero_description" / "urdf" / "nero_with_gripper.urdf")
_DEFAULT_HOME_JSON = _NERO_DIR / "home_joints.json"

if _DEFAULT_HOME_JSON.exists():
    with open(_DEFAULT_HOME_JSON) as _f:
        _DEFAULT_HOME_JOINTS = json.load(_f)["home_joints_rad"]
else:
    _DEFAULT_HOME_JOINTS = [0.0] * 7


@TeleoperatorConfig.register_subclass("nero_gamepad")
@dataclass
class NeroGamepadConfig(TeleoperatorConfig):
    use_gripper: bool = True
    deadzone: float = 0.1

    x_step_size: float = 0.05
    y_step_size: float = 0.05
    z_step_size: float = 0.05
    wx_step_size: float = 0.2
    wy_step_size: float = 0.2
    wz_step_size: float = 0.2
    gripper_step: float = 5
    speed_factor: float = 0.25

    max_joint_step: float = 0.1
    reject_threshold: float = 0.3
    ema_alpha: float = 0.2

    urdf_path: str = _DEFAULT_URDF
    base_link_name: str = "base_link"
    target_link_name: str = "gripper_base"
    home_joint_angles: list[float] = field(default_factory=lambda: list(_DEFAULT_HOME_JOINTS))
