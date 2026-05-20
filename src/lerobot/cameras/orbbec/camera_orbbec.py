import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

import pyorbbecsdk
from pyorbbecsdk import OBFormat, OBPermissionType, OBPropertyID, OBSensorType

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from ..utils import get_cv2_rotation
from .configuration_orbbec import ColorMode, OrbbecCameraConfig

logger = logging.getLogger(__name__)


class OrbbecCamera(Camera):
    def __init__(self, config: OrbbecCameraConfig):
        super().__init__(config)
        self.config = config
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s
        self.enable_depth = config.enable_depth
        self.auto_exposure = config.auto_exposure
        self.auto_white_balance = config.auto_white_balance

        self.pipeline: pyorbbecsdk.Pipeline | None = None
        self.device: pyorbbecsdk.Device | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        sn = self.config.serial_number or "default"
        return f"OrbbecCamera(SN={sn})"

    @property
    def is_connected(self) -> bool:
        return self.pipeline is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        found = []
        try:
            ctx = pyorbbecsdk.Context()
            device_list = ctx.query_devices()
            for i in range(device_list.get_count()):
                info = device_list.get_device_by_index(i).get_device_info()
                found.append({
                    "name": info.get_name(),
                    "type": "orbbec",
                    "vid": f"0x{info.get_vid():X}",
                    "pid": f"0x{info.get_pid():04X}",
                    "sn": info.get_serial_number(),
                })
        except Exception as e:
            logger.warning(f"Failed to enumerate Orbbec cameras: {e}")
        return found

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        if self.config.serial_number:
            ctx = pyorbbecsdk.Context()
            device_list = ctx.query_devices()
            device = None
            for i in range(device_list.get_count()):
                dev = device_list.get_device_by_index(i)
                if dev.get_device_info().get_serial_number() == self.config.serial_number:
                    device = dev
                    break
            if device is None:
                raise ConnectionError(f"Orbbec device SN={self.config.serial_number} not found")
            self.pipeline = pyorbbecsdk.Pipeline(device)
        else:
            self.pipeline = pyorbbecsdk.Pipeline()
        self.device = self.pipeline.get_device()

        self._configure_device()

        config = pyorbbecsdk.Config()
        self._configure_color_stream(config)
        if self.enable_depth:
            self._configure_depth_stream(config)

        self.pipeline.start(config)
        logger.info(f"{self} pipeline started.")

        self._start_read_thread()

        if warmup and self.warmup_s > 0:
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                try:
                    self.async_read(timeout_ms=self.warmup_s * 1000)
                except TimeoutError:
                    pass
                time.sleep(0.1)
            with self.frame_lock:
                if self.latest_frame is None:
                    self._cleanup()
                    raise ConnectionError(f"{self} failed to capture frames during warmup.")

        logger.info(f"{self} connected.")

    def _configure_device(self) -> None:
        if self.device is None:
            return
        if self.auto_exposure and self.device.is_property_supported(
            OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
            OBPermissionType.PERMISSION_READ_WRITE,
        ):
            self.device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, True)

        if self.auto_white_balance and self.device.is_property_supported(
            OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL,
            OBPermissionType.PERMISSION_READ_WRITE,
        ):
            self.device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, True)

    def _configure_color_stream(self, config: pyorbbecsdk.Config) -> None:
        if self.pipeline is None:
            raise DeviceNotConnectedError(f"{self} pipeline is not initialized")

        color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)

        width = self.width or 1280
        height = self.height or 720
        fps = self.fps or 30

        color_profile = color_profiles.get_video_stream_profile(width, height, OBFormat.RGB, fps)
        if color_profile is None:
            color_profile = color_profiles.get_default_video_stream_profile()
            logger.warning(
                f"{self} requested {width}x{height}@{fps}fps RGB not available, using default profile."
            )

        config.enable_stream(color_profile)

    def _configure_depth_stream(self, config: pyorbbecsdk.Config) -> None:
        if self.pipeline is None:
            raise DeviceNotConnectedError(f"{self} pipeline is not initialized")

        depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_video_stream_profile(640, 576, OBFormat.Y16, 30)
        if depth_profile is None:
            depth_profile = depth_profiles.get_default_video_stream_profile()
        config.enable_stream(depth_profile)

    def _read_from_hardware(self) -> NDArray[Any]:
        if self.pipeline is None:
            raise DeviceNotConnectedError(f"{self} pipeline is not initialized")

        frames = self.pipeline.wait_for_frames(1000)
        if frames is None:
            raise RuntimeError(f"{self} wait_for_frames timed out.")

        color_frame = frames.get_color_frame()
        if color_frame is None:
            raise RuntimeError(f"{self} no color frame in frameset.")

        color_data = np.asanyarray(color_frame.get_data())
        color_image = np.copy(color_data.reshape((color_frame.get_height(), color_frame.get_width(), 3)))
        return color_image

    def _postprocess_image(self, image: NDArray[Any]) -> NDArray[Any]:
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid color mode '{self.color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        h, w, c = image.shape
        expected_h = self.capture_height if hasattr(self, "capture_height") and self.capture_height else h
        expected_w = self.capture_width if hasattr(self, "capture_width") and self.capture_width else w

        if h != expected_h or w != expected_w:
            raise RuntimeError(
                f"{self} frame size ({w}x{h}) != expected ({expected_w}x{expected_h})."
            )

        if c != 3:
            raise RuntimeError(f"{self} frame channels={c} != 3.")

        processed = image
        if self.color_mode == ColorMode.BGR:
            processed = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed = cv2.rotate(processed, self.rotation)

        if self.config.crop_width and self.config.crop_height:
            h, w = processed.shape[:2]
            crop_h, crop_w = self.config.crop_height, self.config.crop_width
            y0 = self.config.crop_y_offset if self.config.crop_y_offset is not None else (h - crop_h) // 2
            x0 = self.config.crop_x_offset if self.config.crop_x_offset is not None else (w - crop_w) // 2
            processed = processed[y0:y0+crop_h, x0:x0+crop_w]

        return processed

    def _read_loop(self) -> None:
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event not initialized.")

        failure_count = 0
        while not self.stop_event.is_set():
            try:
                raw_frame = self._read_from_hardware()
                processed_frame = self._postprocess_image(raw_frame)
                capture_time = time.perf_counter()

                with self.frame_lock:
                    self.latest_frame = processed_frame
                    self.latest_timestamp = capture_time
                self.new_frame_event.set()
                failure_count = 0

            except DeviceNotConnectedError:
                break
            except Exception as e:
                failure_count += 1
                if failure_count <= 10:
                    logger.warning(f"Error reading frame in background thread for {self}: {e}")
                else:
                    raise RuntimeError(f"{self} exceeded max consecutive read failures.") from e

    def _start_read_thread(self) -> None:
        self._stop_read_thread()
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()
        time.sleep(0.1)

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self.stop_event = None
        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    @check_if_not_connected
    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        if color_mode is not None:
            logger.warning(
                f"{self} read() color_mode parameter is deprecated."
            )
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        self.new_frame_event.clear()
        return self.async_read(timeout_ms=10000)

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"Timed out waiting for frame from {self} after {timeout_ms} ms. "
                f"Read thread alive: {self.thread.is_alive()}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        with self.frame_lock:
            frame = self.latest_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )

        return frame

    def _cleanup(self) -> None:
        if self.thread is not None:
            self._stop_read_thread()
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
            self.device = None
        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    def disconnect(self) -> None:
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")
        self._cleanup()
        logger.info(f"{self} disconnected.")
