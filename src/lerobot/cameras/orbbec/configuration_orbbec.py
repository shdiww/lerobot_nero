from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode, Cv2Rotation

__all__ = ["OrbbecCameraConfig", "ColorMode", "Cv2Rotation"]


@CameraConfig.register_subclass("orbbec")
@dataclass
class OrbbecCameraConfig(CameraConfig):
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 2
    enable_depth: bool = False
    auto_exposure: bool = True
    auto_white_balance: bool = True
    serial_number: str | None = None
    crop_width: int | None = None
    crop_height: int | None = None
    crop_x_offset: int | None = None
    crop_y_offset: int | None = None

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)
