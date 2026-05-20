"""测试 7: Orbbec LibUVC 后端 vs OpenNI/libusb 后端

测试两种方式打开 Orbbec Femto Bolt:
  方式 A: SDK 默认配置 (Auto -> LibUVC for Femto Bolt)，不设 ORBBEC_SDK_DISABLE_UVC
  方式 B: ORBBEC_SDK_DISABLE_UVC=1，使用 OpenNI/libusb 私有协议

目的: 确定 LibUVC 后端是否可用，若不可用则回退到 OpenNI/libusb
"""
import os
import subprocess
import sys
import time


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
            status = "✅" if val >= 128 else "⚠️  (推荐 >= 128)"
            print(f"  当前值: {val} MB {status}")
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


LIBUVC_TEST_CODE = '''
import sys
import time
import cv2
import numpy as np
import pyorbbecsdk
from pyorbbecsdk import OBFormat, OBPermissionType, OBPropertyID, OBSensorType


def run():
    print("  [1] 枚举 Orbbec 设备...")
    ctx = pyorbbecsdk.Context()
    device_list = ctx.query_devices()
    count = device_list.get_count()
    print(f"  发现 {count} 个 Orbbec 设备")

    if count == 0:
        print("  [FAIL] 未发现 Orbbec 设备！")
        return False

    for i in range(count):
        dev = device_list.get_device_by_index(i)
        info = dev.get_device_info()
        print(f"    设备 {i}: {info.get_name()}, SN={info.get_serial_number()}, "
              f"VID=0x{info.get_vid():X}, PID=0x{info.get_pid():04X}")

    print("  [2] 创建 Pipeline (SDK 默认配置，Femto Bolt 应使用 LibUVC 后端)...")
    pipeline = pyorbbecsdk.Pipeline()
    device = pipeline.get_device()
    dev_info = device.get_device_info()
    print(f"    已打开: {dev_info.get_name()}, SN={dev_info.get_serial_number()}")

    if device.is_property_supported(
        OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
        OBPermissionType.PERMISSION_READ_WRITE,
    ):
        device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, True)
        print("    自动曝光: ON")

    if device.is_property_supported(
        OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL,
        OBPermissionType.PERMISSION_READ_WRITE,
    ):
        device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, True)
        print("    自动白平衡: ON")

    print("  [3] 配置彩色流 1280x720 RGB@30fps...")
    config = pyorbbecsdk.Config()
    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = color_profiles.get_video_stream_profile(1280, 720, OBFormat.RGB, 30)
    if color_profile is None:
        color_profile = color_profiles.get_default_video_stream_profile()
        print("    [WARN] 1280x720@30fps 不可用，使用默认配置")
    config.enable_stream(color_profile)

    print("  [4] 启动 Pipeline...")
    t0 = time.time()
    pipeline.start(config)
    t1 = time.time()
    print(f"    Pipeline 启动耗时: {(t1 - t0) * 1000:.0f} ms")

    print("  [5] 热身: 丢弃前 30 帧...")
    for i in range(30):
        frames = pipeline.wait_for_frames(1000)
        if frames is None:
            print(f"    [WARN] 热身帧 {i + 1} 超时")

    print("  [6] 读取 10 帧并测量延迟...")
    latencies = []
    for i in range(10):
        t_start = time.perf_counter()
        frames = pipeline.wait_for_frames(5000)
        t_end = time.perf_counter()

        if frames is None:
            print(f"    帧 {i + 1}: [TIMEOUT]")
            continue

        color_frame = frames.get_color_frame()
        if color_frame is None:
            print(f"    帧 {i + 1}: [NO COLOR FRAME]")
            continue

        latency_ms = (t_end - t_start) * 1000
        latencies.append(latency_ms)
        print(f"    帧 {i + 1}: {latency_ms:.1f} ms")

    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"    平均: {avg:.1f} ms, 最小: {min(latencies):.1f} ms, 最大: {max(latencies):.1f} ms")

    frames = pipeline.wait_for_frames(5000)
    if frames is not None:
        color_frame = frames.get_color_frame()
        if color_frame is not None:
            data = np.asanyarray(color_frame.get_data())
            image = np.copy(data.reshape((color_frame.get_height(), color_frame.get_width(), 3)))
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            save_path = "/tmp/orbbec_libuvc_test.png"
            cv2.imwrite(save_path, bgr)
            print(f"  [7] 保存图片: {save_path} ({image.shape[1]}x{image.shape[0]})")

    print("  [8] 停止 Pipeline...")
    pipeline.stop()
    print("  [OK] LibUVC 后端测试成功！")
    return True


try:
    success = run()
    sys.exit(0 if success else 1)
except Exception as e:
    print(f"  [FAIL] LibUVC 后端测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''

OPENNI_TEST_CODE = '''
import sys
import time
import cv2
import numpy as np
import os
import pyorbbecsdk
from pyorbbecsdk import OBFormat, OBPermissionType, OBPropertyID, OBSensorType


def run():
    val = os.environ.get("ORBBEC_SDK_DISABLE_UVC", "未设置")
    print(f"  [1] ORBBEC_SDK_DISABLE_UVC = {val}")

    print("  [2] 枚举 Orbbec 设备...")
    ctx = pyorbbecsdk.Context()
    device_list = ctx.query_devices()
    count = device_list.get_count()
    print(f"  发现 {count} 个 Orbbec 设备")

    if count == 0:
        print("  [FAIL] 未发现 Orbbec 设备！")
        return False

    for i in range(count):
        dev = device_list.get_device_by_index(i)
        info = dev.get_device_info()
        print(f"    设备 {i}: {info.get_name()}, SN={info.get_serial_number()}, "
              f"VID=0x{info.get_vid():X}, PID=0x{info.get_pid():04X}")

    print("  [3] 创建 Pipeline (OpenNI/libusb 模式)...")
    pipeline = pyorbbecsdk.Pipeline()
    device = pipeline.get_device()
    dev_info = device.get_device_info()
    print(f"    已打开: {dev_info.get_name()}, SN={dev_info.get_serial_number()}")

    if device.is_property_supported(
        OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
        OBPermissionType.PERMISSION_READ_WRITE,
    ):
        device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, True)
        print("    自动曝光: ON")

    if device.is_property_supported(
        OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL,
        OBPermissionType.PERMISSION_READ_WRITE,
    ):
        device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, True)
        print("    自动白平衡: ON")

    print("  [4] 配置彩色流 1280x720 RGB@30fps...")
    config = pyorbbecsdk.Config()
    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = color_profiles.get_video_stream_profile(1280, 720, OBFormat.RGB, 30)
    if color_profile is None:
        color_profile = color_profiles.get_default_video_stream_profile()
        print("    [WARN] 1280x720@30fps 不可用，使用默认配置")
    config.enable_stream(color_profile)

    print("  [5] 启动 Pipeline...")
    t0 = time.time()
    pipeline.start(config)
    t1 = time.time()
    print(f"    Pipeline 启动耗时: {(t1 - t0) * 1000:.0f} ms")

    print("  [6] 热身: 丢弃前 30 帧...")
    for i in range(30):
        frames = pipeline.wait_for_frames(1000)
        if frames is None:
            print(f"    [WARN] 热身帧 {i + 1} 超时")

    print("  [7] 读取 10 帧并测量延迟...")
    latencies = []
    for i in range(10):
        t_start = time.perf_counter()
        frames = pipeline.wait_for_frames(5000)
        t_end = time.perf_counter()

        if frames is None:
            print(f"    帧 {i + 1}: [TIMEOUT]")
            continue

        color_frame = frames.get_color_frame()
        if color_frame is None:
            print(f"    帧 {i + 1}: [NO COLOR FRAME]")
            continue

        latency_ms = (t_end - t_start) * 1000
        latencies.append(latency_ms)
        print(f"    帧 {i + 1}: {latency_ms:.1f} ms")

    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"    平均: {avg:.1f} ms, 最小: {min(latencies):.1f} ms, 最大: {max(latencies):.1f} ms")

    frames = pipeline.wait_for_frames(5000)
    if frames is not None:
        color_frame = frames.get_color_frame()
        if color_frame is not None:
            data = np.asanyarray(color_frame.get_data())
            image = np.copy(data.reshape((color_frame.get_height(), color_frame.get_width(), 3)))
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            save_path = "/tmp/orbbec_openni_test.png"
            cv2.imwrite(save_path, bgr)
            print(f"  [8] 保存图片: {save_path} ({image.shape[1]}x{image.shape[0]})")

    print("  [9] 停止 Pipeline...")
    pipeline.stop()
    print("  [OK] OpenNI/libusb 后端测试成功！")
    return True


try:
    success = run()
    sys.exit(0 if success else 1)
except Exception as e:
    print(f"  [FAIL] OpenNI/libusb 后端测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''


def test_libuvc_backend():
    print("\n" + "=" * 60)
    print("[方式 A] 测试 LibUVC 后端 (SDK 默认配置，Femto Bolt 默认用 LibUVC)")
    print("         不设置 ORBBEC_SDK_DISABLE_UVC")
    print("=" * 60)

    env = os.environ.copy()
    env.pop("ORBBEC_SDK_DISABLE_UVC", None)

    result = subprocess.run(
        [sys.executable, "-c", LIBUVC_TEST_CODE],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    print(result.stdout)
    if result.stderr:
        for line in result.stderr.splitlines():
            stripped = line.strip().lower()
            if any(k in stripped for k in ("warn", "error", "uvc", "fail", "libuvc")):
                print(f"  [STDERR] {line}")

    return result.returncode == 0


def test_openni_backend():
    print("\n" + "=" * 60)
    print("[方式 B] 测试 OpenNI/libusb 后端 (ORBBEC_SDK_DISABLE_UVC=1)")
    print("=" * 60)

    env = os.environ.copy()
    env["ORBBEC_SDK_DISABLE_UVC"] = "1"

    result = subprocess.run(
        [sys.executable, "-c", OPENNI_TEST_CODE],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    print(result.stdout)
    if result.stderr:
        for line in result.stderr.splitlines():
            stripped = line.strip().lower()
            if any(k in stripped for k in ("warn", "error", "uvc", "fail", "libuvc")):
                print(f"  [STDERR] {line}")

    return result.returncode == 0


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Orbbec LibUVC vs OpenNI/libusb 后端测试               ║")
    print("║  设备: Femto Bolt (默认 LibUVC 后端)                   ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    check_usb_topology()

    libuvc_ok = test_libuvc_backend()
    openni_ok = test_openni_backend()

    print("\n" + "=" * 60)
    print("[总结]")
    print("=" * 60)

    if libuvc_ok:
        print("  ✅ LibUVC 后端: 可用 (用户态绕过内核 uvcvideo)")
    else:
        print("  ❌ LibUVC 后端: 不可用 (可能与 uvcvideo 内核驱动冲突)")

    if openni_ok:
        print("  ✅ OpenNI/libusb 后端: 可用 (ORBBEC_SDK_DISABLE_UVC=1)")
    else:
        print("  ❌ OpenNI/libusb 后端: 不可用")

    print()
    if libuvc_ok:
        print("  推荐配置: 不需要设置 ORBBEC_SDK_DISABLE_UVC")
        print("  Femto Bolt 默认使用 LibUVC，与 RealSense 的 uvcvideo 不冲突")
    elif openni_ok:
        print("  推荐配置: 设置 ORBBEC_SDK_DISABLE_UVC=1")
        print("  Orbbec 使用 OpenNI/libusb，RealSense 独占 uvcvideo")
    else:
        print("  ⚠️  两种后端均不可用，请检查:")
        print("    1. USB 线缆是否连接 (USB 3.0 蓝色端口)")
        print("    2. udev 规则是否安装 (99-obsensor-libusb.rules)")
        print("    3. usbfs_memory_mb 是否 >= 128")
        print("    4. 设备是否被其他进程占用")


if __name__ == "__main__":
    main()
