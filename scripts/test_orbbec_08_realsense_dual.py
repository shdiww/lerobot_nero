"""测试 8: Orbbec Femto Bolt + RealSense 双相机并发测试

验证两个相机能否同时工作无堵塞:
  - Orbbec: ORBBEC_SDK_DISABLE_UVC=1 -> OpenNI/libusb (避免与 uvcvideo 冲突)
  - RealSense: pyrealsense2 -> uvcvideo/V4L2

测试内容:
  1. 环境检测 (USB 拓扑、usbfs_memory_mb、uvcvideo 模块)
  2. 枚举所有相机
  3. 同时启动两个相机的后台读线程
  4. 并发采集 10 秒，统计帧率、延迟、丢帧
  5. 报告结果
"""

import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

os.environ["ORBBEC_SDK_DISABLE_UVC"] = "1"

import cv2
import numpy as np
import pyorbbecsdk
import pyrealsense2 as rs
from pyorbbecsdk import OBFormat, OBPermissionType, OBPropertyID, OBSensorType


@dataclass
class CameraStats:
    name: str = ""
    frame_count: int = 0
    timeout_count: int = 0
    error_count: int = 0
    latencies_ms: list = field(default_factory=list)
    frame_timestamps: list = field(default_factory=list)
    first_frame_time: float | None = None
    last_frame_time: float | None = None

class OrbbecReader:
    def __init__(self, width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.device = None
        self.thread = None
        self.stop_event = threading.Event()
        self.latest_frame = None
        self.latest_timestamp = 0.0
        self.frame_lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.stats = CameraStats(name="Orbbec Femto Bolt")
        self.sn = None

    def connect(self):
        ctx = pyorbbecsdk.Context()
        device_list = ctx.query_devices()
        count = device_list.get_count()
        if count == 0:
            raise ConnectionError("未发现 Orbbec 设备")

        dev = None
        for i in range(count):
            d = device_list.get_device_by_index(i)
            info = d.get_device_info()
            print(f"    Orbbec 设备 {i}: {info.get_name()}, SN={info.get_serial_number()}")
            if dev is None:
                dev = d
                self.sn = info.get_serial_number()

        self.pipeline = pyorbbecsdk.Pipeline(dev)
        self.device = self.pipeline.get_device()

        if self.device.is_property_supported(
            OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
            OBPermissionType.PERMISSION_READ_WRITE,
        ):
            self.device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, True)

        if self.device.is_property_supported(
            OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL,
            OBPermissionType.PERMISSION_READ_WRITE,
        ):
            self.device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, True)

        config = pyorbbecsdk.Config()
        color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profiles.get_video_stream_profile(
            self.width, self.height, OBFormat.RGB, self.fps
        )
        if color_profile is None:
            color_profile = color_profiles.get_default_video_stream_profile()
            print(f"    [WARN] {self.width}x{self.height}@{self.fps}fps 不可用，使用默认配置")
        config.enable_stream(color_profile)

        self.pipeline.start(config)
        print(f"    Orbbec Pipeline 启动成功 (SN={self.sn})")

    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._read_loop, name="orbbec_read_loop", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self.thread = None

    def disconnect(self):
        self.stop()
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
            self.device = None

    def _read_loop(self):
        while not self.stop_event.is_set():
            try:
                t0 = time.perf_counter()
                frames = self.pipeline.wait_for_frames(1000)
                if frames is None:
                    self.stats.timeout_count += 1
                    continue

                color_frame = frames.get_color_frame()
                if color_frame is None:
                    self.stats.error_count += 1
                    continue

                data = np.asanyarray(color_frame.get_data())
                image = np.copy(data.reshape((color_frame.get_height(), color_frame.get_width(), 3)))
                bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                t1 = time.perf_counter()
                latency = (t1 - t0) * 1000

                with self.frame_lock:
                    self.latest_frame = bgr
                    self.latest_timestamp = t1
                self.new_frame_event.set()

                self.stats.frame_count += 1
                self.stats.latencies_ms.append(latency)
                self.stats.frame_timestamps.append(t1)
                if self.stats.first_frame_time is None:
                    self.stats.first_frame_time = t1
                self.stats.last_frame_time = t1

            except Exception as e:
                self.stats.error_count += 1
                if self.stats.error_count <= 3:
                    print(f"    [Orbbec 读帧错误] {e}")

    def get_latest_frame(self):
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy(), self.latest_timestamp
            return None, 0.0

class RealSenseReader:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.rs_pipeline = None
        self.rs_profile = None
        self.serial_number = None
        self.device_name = None
        self.thread = None
        self.stop_event = threading.Event()
        self.latest_frame = None
        self.latest_timestamp = 0.0
        self.frame_lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.stats = CameraStats(name="RealSense")

    def connect(self):
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise ConnectionError("未发现 RealSense 设备")

        dev = devices[0]
        self.serial_number = dev.get_info(rs.camera_info.serial_number)
        self.device_name = dev.get_info(rs.camera_info.name)
        for i, d in enumerate(devices):
            print(
                f"    RealSense 设备 {i}: {d.get_info(rs.camera_info.name)}, "
                f"SN={d.get_info(rs.camera_info.serial_number)}"
            )

        self.rs_pipeline = rs.pipeline()
        rs_config = rs.config()
        rs.config.enable_device(rs_config, self.serial_number)

        supported = self._find_best_profile(dev)
        w, h, f = supported
        self.width, self.height, self.fps = w, h, f
        print(f"    使用分辨率: {w}x{h}@{f}fps")

        rs_config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, f)

        try:
            self.rs_profile = self.rs_pipeline.start(rs_config)
        except RuntimeError as e:
            self.rs_pipeline = None
            raise ConnectionError(f"RealSense Pipeline 启动失败: {e}") from e

        print(f"    RealSense Pipeline 启动成功 ({self.device_name}, SN={self.serial_number})")

    def _find_best_profile(self, dev):
        preferred = [(640, 480, 30), (640, 480, 15), (424, 240, 30), (424, 240, 15)]

        sensors = dev.query_sensors()
        for sensor in sensors:
            profiles = sensor.get_stream_profiles()
            available = set()
            for p in profiles:
                if p.is_video_stream_profile():
                    vp = p.as_video_stream_profile()
                    if vp.stream_type() == rs.stream.color:
                        available.add((vp.width(), vp.height(), vp.fps()))

            for w, h, f in preferred:
                if (w, h, f) in available:
                    return (w, h, f)

            if available:
                for p in sorted(available, key=lambda x: x[0] * x[1]):
                    return p

        return (640, 480, 30)

    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._read_loop, name="realsense_read_loop", daemon=True
        )
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self.thread = None

    def disconnect(self):
        self.stop()
        if self.rs_pipeline:
            self.rs_pipeline.stop()
            self.rs_pipeline = None
            self.rs_profile = None

    def _read_loop(self):
        while not self.stop_event.is_set():
            try:
                t0 = time.perf_counter()
                ret, frames = self.rs_pipeline.try_wait_for_frames(timeout_ms=1000)
                if not ret or frames is None:
                    self.stats.timeout_count += 1
                    continue

                color_frame = frames.get_color_frame()
                if color_frame is None:
                    self.stats.error_count += 1
                    continue

                color_data = np.asanyarray(color_frame.get_data())
                bgr = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)

                t1 = time.perf_counter()
                latency = (t1 - t0) * 1000

                with self.frame_lock:
                    self.latest_frame = bgr
                    self.latest_timestamp = t1
                self.new_frame_event.set()

                self.stats.frame_count += 1
                self.stats.latencies_ms.append(latency)
                self.stats.frame_timestamps.append(t1)
                if self.stats.first_frame_time is None:
                    self.stats.first_frame_time = t1
                self.stats.last_frame_time = t1

            except Exception as e:
                self.stats.error_count += 1
                if self.stats.error_count <= 3:
                    print(f"    [RealSense 读帧错误] {e}")

    def get_latest_frame(self):
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy(), self.latest_timestamp
            return None, 0.0

def check_usb_topology():
    print("=" * 60)
    print("[环境检测]")
    print("=" * 60)

    print("\n--- USB 拓扑 (lsusb -t) ---")
    result = subprocess.run(["lsusb", "-t"], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"  无法获取: {result.stderr}")

    print("--- uvcvideo 内核模块 ---")
    result = subprocess.run(["lsmod"], capture_output=True, text=True)
    uvc_lines = [l for l in result.stdout.splitlines() if "uvc" in l]
    if uvc_lines:
        for l in uvc_lines:
            print(f"  {l}")
    else:
        print("  uvcvideo 模块未加载")

    print("--- usbfs_memory_mb ---")
    try:
        with open("/sys/module/usbcore/parameters/usbfs_memory_mb") as f:
            val = int(f.read().strip())
            status = "OK" if val >= 128 else "WARN (推荐 >= 128)"
            print(f"  当前值: {val} MB [{status}]")
    except Exception:
        print("  无法读取")

    print("--- ORBBEC_SDK_DISABLE_UVC ---")
    val = os.environ.get("ORBBEC_SDK_DISABLE_UVC", "未设置")
    print(f"  当前值: {val}")

    print("\n--- /dev/video* 设备 ---")
    result = subprocess.run(["ls", "-la", "/dev/video*"], capture_output=True, text=True)
    if result.returncode == 0:
        for line in result.stdout.strip().splitlines():
            print(f"  {line}")
    else:
        print("  无 /dev/video* 设备")


def print_stats(reader, test_duration_s):
    stats = reader.stats
    name = stats.name

    print(f"\n  [{name}] 统计:")
    print(f"    总帧数: {stats.frame_count}")
    print(f"    超时次数: {stats.timeout_count}")
    print(f"    错误次数: {stats.error_count}")

    if stats.first_frame_time and stats.last_frame_time:
        actual_duration = stats.last_frame_time - stats.first_frame_time
        if actual_duration > 0:
            effective_fps = stats.frame_count / actual_duration
            print(f"    实际帧率: {effective_fps:.1f} fps")
            print(f"    实际采集时长: {actual_duration:.1f} s")

    if stats.latencies_ms:
        avg = sum(stats.latencies_ms) / len(stats.latencies_ms)
        sorted_l = sorted(stats.latencies_ms)
        p50 = sorted_l[len(sorted_l) // 2]
        p95 = sorted_l[int(len(sorted_l) * 0.95)]
        print(
            f"    读帧延迟: avg={avg:.1f}ms, p50={p50:.1f}ms, p95={p95:.1f}ms, "
            f"min={min(stats.latencies_ms):.1f}ms, max={max(stats.latencies_ms):.1f}ms"
        )

    if stats.timeout_count > 0 or stats.error_count > 0:
        total_issues = stats.timeout_count + stats.error_count
        issue_rate = total_issues / max(stats.frame_count + total_issues, 1) * 100
        print(f"    问题率: {issue_rate:.1f}% ({total_issues}/{stats.frame_count + total_issues})")


def compute_inter_camera_stats(orbbec_reader, realsense_reader):
    orb_ts = orbbec_reader.stats.frame_timestamps
    rs_ts = realsense_reader.stats.frame_timestamps

    if not orb_ts or not rs_ts:
        print("\n  [两相机帧时间差] 数据不足，无法计算")
        return

    diffs = []
    oi, ri = 0, 0
    while oi < len(orb_ts) and ri < len(rs_ts):
        diff = abs(orb_ts[oi] - rs_ts[ri]) * 1000
        diffs.append(diff)
        if orb_ts[oi] < rs_ts[ri]:
            oi += 1
        else:
            ri += 1

    if diffs:
        avg_diff = sum(diffs) / len(diffs)
        sorted_diffs = sorted(diffs)
        p50 = sorted_diffs[len(sorted_diffs) // 2]
        p95 = sorted_diffs[int(len(sorted_diffs) * 0.95)]
        print(
            f"\n  [两相机帧时间差]"
            f"\n    avg={avg_diff:.1f}ms, p50={p50:.1f}ms, p95={p95:.1f}ms, "
            f"min={min(diffs):.1f}ms, max={max(diffs):.1f}ms"
        )


def save_sample_frames(orbbec_reader, realsense_reader):
    orb_frame, _ = orbbec_reader.get_latest_frame()
    rs_frame, _ = realsense_reader.get_latest_frame()

    if orb_frame is not None:
        path = "/tmp/dual_test_orbbec.png"
        cv2.imwrite(path, orb_frame)
        print(f"\n  [Orbbec] 保存样本帧: {path} ({orb_frame.shape[1]}x{orb_frame.shape[0]})")

    if rs_frame is not None:
        path = "/tmp/dual_test_realsense.png"
        cv2.imwrite(path, rs_frame)
        print(f"  [RealSense] 保存样本帧: {path} ({rs_frame.shape[1]}x{rs_frame.shape[0]})")

    if orb_frame is not None and rs_frame is not None:
        h1, w1 = orb_frame.shape[:2]
        h2, w2 = rs_frame.shape[:2]
        target_h = min(h1, h2)
        orb_resized = orb_frame
        rs_resized = rs_frame
        if h1 != target_h:
            scale = target_h / h1
            orb_resized = cv2.resize(orb_frame, (int(w1 * scale), target_h))
        if h2 != target_h:
            scale = target_h / h2
            rs_resized = cv2.resize(rs_frame, (int(w2 * scale), target_h))

        combined = np.hstack([orb_resized, rs_resized])
        cv2.putText(combined, "Orbbec", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(
            combined,
            "RealSense",
            (orb_resized.shape[1] + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        path = "/tmp/dual_test_combined.png"
        cv2.imwrite(path, combined)
        print(f"  [Combined] 拼接图: {path}")


def main():
    print("=" * 60)
    print("  Orbbec + RealSense 双相机并发测试")
    print("  Orbbec: ORBBEC_SDK_DISABLE_UVC=1 (OpenNI/libusb)")
    print("  RealSense: uvcvideo/V4L2")
    print("=" * 60 + "\n")

    check_usb_topology()

    test_duration_s = 10

    orbbec_reader = OrbbecReader(width=1280, height=720, fps=30)
    realsense_reader = RealSenseReader(width=640, height=480, fps=30)

    print("\n" + "=" * 60)
    print("[步骤 1] 连接 Orbbec 相机...")
    print("=" * 60)
    try:
        orbbec_reader.connect()
    except Exception as e:
        print(f"[FAIL] Orbbec 连接失败: {e}")
        print("请检查: 1) USB 连接  2) udev 规则  3) 设备未被占用")
        return

    print("\n" + "=" * 60)
    print("[步骤 2] 连接 RealSense 相机...")
    print("=" * 60)
    try:
        realsense_reader.connect()
    except Exception as e:
        print(f"[FAIL] RealSense 连接失败: {e}")
        print("请检查: 1) USB 连接  2) uvcvideo 模块  3) 设备未被占用")
        orbbec_reader.disconnect()
        return

    print("\n" + "=" * 60)
    print("[步骤 3] 热身: 等待两个相机稳定 (3 秒)...")
    print("=" * 60)
    orbbec_reader.start()
    realsense_reader.start()
    time.sleep(3.0)

    orb_warmup = orbbec_reader.stats.frame_count
    rs_warmup = realsense_reader.stats.frame_count
    print(f"  Orbbec 热身帧数: {orb_warmup}")
    print(f"  RealSense 热身帧数: {rs_warmup}")

    if orb_warmup == 0:
        print("[FAIL] Orbbec 热身期间无帧，请检查设备")
        orbbec_reader.disconnect()
        realsense_reader.disconnect()
        return
    if rs_warmup == 0:
        print("[FAIL] RealSense 热身期间无帧，请检查设备")
        orbbec_reader.disconnect()
        realsense_reader.disconnect()
        return

    orbbec_reader.stats = CameraStats(name="Orbbec Femto Bolt")
    realsense_reader.stats = CameraStats(name="RealSense")

    print("\n" + "=" * 60)
    print(f"[步骤 4] 并发采集测试 ({test_duration_s} 秒)...")
    print("=" * 60)

    sample_interval = 1.0 / 30
    start_time = time.time()
    poll_count = 0

    while time.time() - start_time < test_duration_s:
        poll_count += 1

        orb_frame, orb_ts = orbbec_reader.get_latest_frame()
        rs_frame, rs_ts = realsense_reader.get_latest_frame()

        if poll_count % 30 == 0:
            elapsed = time.time() - start_time
            orb_count = orbbec_reader.stats.frame_count
            rs_count = realsense_reader.stats.frame_count
            print(
                f"  [{elapsed:.1f}s] "
                f"Orbbec: {orb_count} 帧, "
                f"RealSense: {rs_count} 帧"
            )

        time.sleep(sample_interval)

    end_time = time.time()
    actual_duration = end_time - start_time

    print("\n" + "=" * 60)
    print("[步骤 5] 保存样本帧...")
    print("=" * 60)
    save_sample_frames(orbbec_reader, realsense_reader)

    print("\n" + "=" * 60)
    print("[步骤 6] 停止采集...")
    print("=" * 60)
    orbbec_reader.disconnect()
    realsense_reader.disconnect()

    print("\n" + "=" * 60)
    print("[测试结果]")
    print("=" * 60)
    print(f"  测试时长: {actual_duration:.1f} 秒")
    print()

    print_stats(orbbec_reader, actual_duration)
    print_stats(realsense_reader, actual_duration)
    compute_inter_camera_stats(orbbec_reader, realsense_reader)

    print("\n" + "-" * 60)
    print("[诊断]")
    orb_ok = orbbec_reader.stats.frame_count > 0
    rs_ok = realsense_reader.stats.frame_count > 0
    orb_issues = orbbec_reader.stats.timeout_count + orbbec_reader.stats.error_count
    rs_issues = realsense_reader.stats.timeout_count + realsense_reader.stats.error_count

    if orb_ok and rs_ok and orb_issues == 0 and rs_issues == 0:
        print("  双相机并发测试通过！无堵塞、无丢帧、无冲突。")
    elif orb_ok and rs_ok:
        print("  双相机可同时工作，但存在问题:")
        if orb_issues > 0:
            print(f"    Orbbec: {orb_issues} 次超时/错误，可能原因:")
            print(f"      - USB 带宽不足 (尝试降低分辨率/帧率)")
            print(f"      - uvcvideo 驱动冲突 (确保 ORBBEC_SDK_DISABLE_UVC=1)")
        if rs_issues > 0:
            print(f"    RealSense: {rs_issues} 次超时/错误，可能原因:")
            print(f"      - USB 带宽不足 (尝试降低分辨率/帧率)")
            print(f"      - uvcvideo 驱动冲突 (确保 Orbbec 不使用 uvcvideo)")
    else:
        if not orb_ok:
            print("  Orbbec 采集失败，请检查:")
            print("    1. ORBBEC_SDK_DISABLE_UVC=1 是否已设置 (在 import 前)")
            print("    2. USB 3.0 连接是否稳定")
            print("    3. udev 规则 99-obsensor-libusb.rules 是否安装")
            print("    4. usbfs_memory_mb 是否 >= 128")
        if not rs_ok:
            print("  RealSense 采集失败，请检查:")
            print("    1. uvcvideo 内核模块是否加载")
            print("    2. USB 3.0 连接是否稳定")
            print("    3. 设备是否被其他进程占用")
            print("    4. 两个相机是否在不同 USB 控制器上 (lsusb -t)")


if __name__ == "__main__":
    main()
