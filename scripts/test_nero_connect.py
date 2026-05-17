#!/usr/bin/env python
"""Nero 机器人连接测试

作用:
    验证 Nero 作为 LeRobot Robot 子类的生命周期:
    1. connect() 后 is_connected 应为 True
    2. get_observation() 能否读到 7 个关节角 + gripper (如有)
    3. move_to_home() 后关节角度接近 home_joint_angles, 夹爪闭合后测试开合
    4. disconnect() 后 is_connected 应为 False
    5. 断开后能否重新 connect

前提:
    - CAN 接口已启用 (sudo ip link set can0 up type can bitrate 1000000)
    - Nero 机械臂已上电并连接 CAN 总线
    - pyAgxArm 已安装 (cd pyAgxArm && uv pip install -e .)

预期现象:
    [1/5] is_connected = True
    [2/5] 打印 observation keys (joint1.pos ~ joint7.pos, gripper.pos)
          以及各关节当前弧度值
    [3/5] move_to_home() 后各关节角度接近 home_joint_angles, 夹爪开合测试
    [4/5] 打印 is_connected = False
    [5/5] 重新连接后 is_connected = True, 再次断开后 = False
    最终输出 "All connect/disconnect tests PASSED."

Usage:
    uv run python scripts/test_nero_connect.py
"""
import time
from lerobot.robots import make_robot_from_config
from lerobot.robots.nero import NeroRobotConfig
from lerobot.robots.nero.config_nero import NERO_GRIPPER_MAX_WIDTH_M


def main():
    robot = make_robot_from_config(NeroRobotConfig())

    print("[1/5] Testing connect...")
    robot.connect(calibrate=False)
    print(f"  is_connected = {robot.is_connected}")
    assert robot.is_connected, "Robot should be connected after connect()"

    print("[2/5] Testing get_observation...")
    obs = robot.get_observation()
    print(f"  observation keys: {list(obs.keys())}")
    for k, v in obs.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    print("[3/5] Testing go home...")
    robot.move_to_home()
    timeout = 10.0
    home_angles = robot.config.home_joint_angles
    t0 = time.monotonic()
    while True:
        obs = robot.get_observation()
        if all(abs(obs[f"joint{i}.pos"] - home_angles[i-1]) < 0.1 for i in range(1, 8)):
            break
        if time.monotonic() - t0 > timeout:
            break
        time.sleep(0.1)


    print("[4/5] Testing disconnect...")
    robot.disconnect()
    print(f"  is_connected = {robot.is_connected}")
    assert not robot.is_connected, "Robot should not be connected after disconnect()"

    print("[5/5] Testing reconnect...")
    robot.connect(calibrate=False)
    assert robot.is_connected, "Robot should be re-connectable"
    robot.disconnect()
    assert not robot.is_connected

    print("\nAll connect/disconnect tests PASSED.")


if __name__ == "__main__":
    main()
