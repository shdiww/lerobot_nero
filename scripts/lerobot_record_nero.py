#!/usr/bin/env python
"""Nero 专用数据采集脚本.

基于遥操作脚本，增加 LeRobot 数据集录制功能。

数据流:
    Xbox 手柄摇杆/按键
        │
        ▼
    NeroGamepad.get_action() → deltas
        │
        ▼
    teleop.solve_ik_from_deltas() → joint_angles
        │
        ▼
    拼装 action: {"joint1.pos": ..., "gripper.pos": ...}
        │
        ├──→ robot.send_action(action)
        │
        └──→ dataset.add_frame({
                "observation.state": [j1...j7, gripper],
                "observation.images.top": image,
                "action": [j1...j7, gripper],
                "task": single_task,
             })

按键:
    L-stick  → 末端 X/Y 平移
    R-stick  → 末端 Z 平移 / Z 旋转
    D-pad    → 末端 X/Y 旋转
    A        → 夹爪闭合
    B        → 夹爪张开
    Y        → 回 Home 位置
    Back     → E-STOP
    Home     → 连接机械臂
    → (右箭头) → 提前结束当前 episode
    ← (左箭头) → 重录当前 episode
    Esc      → 停止所有录制

Usage:
    uv run python scripts/lerobot_record_nero.py \
        --robot.type=nero --teleop.type=nero_gamepad \
        --use_orbbec_camera=true \
        --display_cameras=true \
        --dataset.repo_id=nero_pick_cup \
        --dataset.num_episodes=50 \
        --dataset.single_task="pick up the cup"
"""

import logging
import os
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import cv2
import numpy as np
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.orbbec import OrbbecCameraConfig  # noqa: F401
from lerobot.common.control_utils import init_keyboard_listener, is_headless
from lerobot.configs import parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import LeRobotDataset, VideoEncodingManager
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.processor import make_default_processors
from lerobot.robots import Robot, RobotConfig, make_robot_from_config, nero  # noqa: F401
from lerobot.robots.nero import Nero
from lerobot.robots.nero.config_nero import NERO_GRIPPER_MAX_WIDTH_M, NERO_JOINT_NAMES
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
    nero_gamepad,
)
from lerobot.teleoperators.nero_gamepad import NeroGamepad
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


@dataclass
class NeroRecordConfig:
    robot: RobotConfig
    teleop: TeleoperatorConfig
    dataset: DatasetRecordConfig
    fps: int = 30
    use_orbbec_camera: bool = False
    display_cameras: bool = False
    display_data: bool = False
    resume: bool = False


def _init_can():
    os.system("sudo ip link set can0 up type can bitrate 1000000 2>/dev/null")
    logger.info("CAN bus initialized (can0, 1Mbps)")


def _sync_ik_from_robot(teleop: NeroGamepad, robot: Nero, retries: int = 10) -> None:
    if teleop.ik_solver is None:
        return

    actual_joints = None
    for attempt in range(retries):
        obs = robot.get_observation()
        if all(f"{name}.pos" in obs for name in NERO_JOINT_NAMES):
            actual_joints = [obs[f"{name}.pos"] for name in NERO_JOINT_NAMES]
            break
        logger.warning(f"get_observation missing joint data (attempt {attempt + 1}/{retries}), retrying...")
        time.sleep(0.5)

    if actual_joints is None:
        logger.warning("Failed to read joint angles from robot, falling back to home")
        actual_joints = list(teleop.config.home_joint_angles)

    teleop.set_joint_angles(actual_joints)
    teleop.ik_solver.set_seed(actual_joints)
    logger.info(f"IK synced to joints: {[f'{v:.3f}' for v in actual_joints]}")


def record_episode_loop(
    teleop: NeroGamepad,
    robot: Nero,
    fps: int,
    events: dict,
    dataset: LeRobotDataset | None = None,
    control_time_s: float | None = None,
    single_task: str | None = None,
    display_cameras: bool = False,
):
    control_interval = 1 / fps
    start_episode_t = time.perf_counter()
    timestamp = 0.0
    going_home = False

    while timestamp < (control_time_s or float("inf")):
        loop_start = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        if events["stop_recording"]:
            break

        raw_action = teleop.get_action()
        teleop_events = teleop.get_teleop_events()

        if teleop_events.get(TeleopEvents.SUCCESS):
            logger.info("Y pressed — moving to Home position")
            going_home = True
            robot.move_to_home()
            _sync_ik_from_robot(teleop, robot)
            teleop.gripper_state = 100.0
            going_home = False
            continue

        if teleop_events.get(TeleopEvents.TERMINATE_EPISODE):
            logger.info("Back pressed — E-STOP, moving to Home then disconnecting")
            robot.move_to_home()
            events["stop_recording"] = True
            break

        if teleop.is_home_requested():
            logger.info("Home button pressed — reconnecting robot")
            if not robot.is_connected:
                robot.connect()
                _sync_ik_from_robot(teleop, robot)

        if going_home:
            dt_s = time.perf_counter() - loop_start
            precise_sleep(max(control_interval - dt_s, 0.0))
            timestamp = time.perf_counter() - start_episode_t
            continue

        delta_x = raw_action["delta_x"]
        delta_y = raw_action["delta_y"]
        delta_z = raw_action["delta_z"]
        delta_wx = raw_action["delta_wx"]
        delta_wy = raw_action["delta_wy"]
        delta_wz = raw_action["delta_wz"]
        delta_gripper = raw_action.get("delta_gripper", 0.0)

        has_input = any(v != 0.0 for v in (delta_x, delta_y, delta_z, delta_wx, delta_wy, delta_wz))

        if has_input:
            teleop.solve_ik_from_deltas(delta_x, delta_y, delta_z, delta_wx, delta_wy, delta_wz)

        if delta_gripper != 0.0:
            teleop.update_gripper_state(delta_gripper)

        action = {}
        for name, val in zip(NERO_JOINT_NAMES, teleop.get_joint_angles(), strict=True):
            action[f"{name}.pos"] = val

        if teleop.config.use_gripper:
            action["gripper.pos"] = NERO_GRIPPER_MAX_WIDTH_M * teleop.gripper_state * 1e-2

        robot.send_action(action)

        if dataset is not None:
            obs = robot.get_observation()

            observation_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
            action_frame = build_dataset_frame(dataset.features, action, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_cameras and robot.cameras:
            for cam_name, cam in robot.cameras.items():
                try:
                    frame_img = cam.read_latest(max_age_ms=2000)
                    if frame_img is not None:
                        frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
                        cv2.imshow(f"Nero - {cam_name}", frame_bgr)
                except (TimeoutError, RuntimeError):
                    pass
            cv2.waitKey(1)

        dt_s = time.perf_counter() - loop_start
        sleep_time_s = control_interval - dt_s
        if sleep_time_s < 0:
            logger.warning(
                f"Record loop running at {1/dt_s:.1f} Hz, target {fps} Hz. "
                "Consider reducing camera resolution or fps."
            )
        precise_sleep(max(sleep_time_s, 0.0))

        timestamp = time.perf_counter() - start_episode_t
        loop_s = time.perf_counter() - loop_start
        print(f"Record loop: {loop_s * 1e3:.1f}ms ({1 / loop_s:.0f} Hz) | ep: {timestamp:.1f}s / {control_time_s}s")
        move_cursor_up(1)


def move_cursor_up(n: int):
    print(f"\033[{n}A", end="")


@parser.wrap()
def record(cfg: NeroRecordConfig):
    init_logging()

    if cfg.use_orbbec_camera:
        cfg.robot.cameras["top"] = OrbbecCameraConfig(
            fps=30, width=1280, height=720,
            color_mode=ColorMode.RGB, warmup_s=2,
            auto_exposure=True, auto_white_balance=True,
        )
        logger.info("Orbbec camera attached as 'top'")

    cfg.robot.motion_mode = "js"
    logger.info("Motion mode forced to 'js' (servo mode) for teleoperation")

    logging.info(pformat(asdict(cfg)))

    _init_can()

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)

    _t, _r, _o = make_default_processors()
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=_t,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=_o,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    logger.info(f"Dataset features: {dataset_features}")

    dataset = None
    listener = None

    try:
        if cfg.resume:
            num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras if num_cameras > 0 else 0,
            )
        else:
            cfg.dataset.stamp_repo_id()
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )

        logger.info(f"Dataset: {cfg.dataset.repo_id}, episodes: {dataset.num_episodes}, frames: {dataset.num_frames}")

        teleop.connect()
        robot.connect()
        _sync_ik_from_robot(teleop, robot)

        listener, events = init_keyboard_listener()

        if not is_headless() and cfg.display_cameras:
            logger.info("Camera display enabled. Press 'q' in camera window or Esc to stop.")

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                logger.info(f"=== Recording episode {recorded_episodes + 1}/{cfg.dataset.num_episodes} ===")

                record_episode_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.dataset.fps,
                    events=events,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_cameras=cfg.display_cameras,
                )

                if not events["stop_recording"] and (
                    recorded_episodes < cfg.dataset.num_episodes - 1 or events["rerecord_episode"]
                ):
                    logger.info(f"--- Reset phase ({cfg.dataset.reset_time_s}s) ---")
                    record_episode_loop(
                        teleop=teleop,
                        robot=robot,
                        fps=cfg.dataset.fps,
                        events=events,
                        dataset=None,
                        control_time_s=cfg.dataset.reset_time_s,
                        display_cameras=cfg.display_cameras,
                    )

                if events["rerecord_episode"]:
                    logger.info("Re-recording episode...")
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1
                logger.info(f"Episode {recorded_episodes} saved. Total: {dataset.num_episodes}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        logger.info("Finalizing dataset...")

        if dataset is not None:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()

        if not is_headless() and listener is not None:
            listener.stop()

        if cfg.display_cameras:
            cv2.destroyAllWindows()

        if dataset is not None:
            logger.info(f"Dataset saved: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
            logger.info(f"Location: {dataset.root}")


def main():
    record()


if __name__ == "__main__":
    main()
