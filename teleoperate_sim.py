"""
Simulated teleoperation: reads teleoperator hardware input and uses it to control a Mujoco simulation.
No physical robot is connected or commanded. Use this to control a robot in simulation.

Example usage:

mjpython -m lerobot.teleoperate_sim \
  --teleop.type=so100_leader \
  --teleop.port=/dev/tty.usbmodemXXXX \
  --teleop.id=my_leader \
  --mjcf_path=path/to/your_robot.xml \
  --display_data=true

执行示例（使用前确认ACM）:
    python -m lerobot.teleoperate_sim --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=my_awesome_leader_arm
开rerun的：
    python -m lerobot.teleoperate_sim --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=my_awesome_leader_arm --display_data True

"""

import logging
import time
import threading
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus
import mujoco
import mujoco.viewer
import numpy as np
import rerun as rr

from lerobot.teleoperators import (
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import _init_rerun

from lerobot.teleoperators import koch_leader, so100_leader, so101_leader  # noqa: F401

# 导入异步录制模块
from async_recording import AsyncRecordingManager


# 自行测量real-sim关节偏置量
def teleop_offset(action: list[float]):
    """
    关节偏置量
    -0.0612
    0.0175
    0.112
    0.0447
    0.119
    0.948
    """
    action[0] -= -0.0612
    action[1] -= 0.0175
    action[2] -= 0.112
    action[3] -= 0.0447
    action[4] -= 0.119
    # 夹爪模型安装孔位与leader不同，但与follower相同，要再偏置0.795
    action[5] -= 0.948 - 0.795
    # return action

@dataclass
class TeleoperateSimConfig:
    """
    执行示例（使用前确认ACM）:
    python -m lerobot.teleoperate_sim --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=my_awesome_leader_arm
    开rerun的：
    python -m lerobot.teleoperate_sim --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=my_awesome_leader_arm --display_data True
    """
    teleop: TeleoperatorConfig
    # 模型路径 - 使用项目内的 scene3.xml
    mjcf_path: str = "/home/b760m/workspace/mujoco/Simulation/SO101/scene2.xml"
    # mjcf_path: str = "/home/b760m/workspace/mujoco/lerobot_so100_sim/so100_sim/so100_6dof/push_cube_loop.xml"
    # mjcf_path: str = "/home/b760m/workspace/mujoco/Simulation/Scene100/myscene1.xml"
    # mjcf_path: str = "/home/b760m/workspace/mujoco/Simulation/SO101/myscene1.xml"
    fps: int = 60
    display_data: bool = False

"""    
import mujoco
import mujoco.viewer as v
model_path = "/home/b760m/workspace/mujoco/Simulation/SO101/scene2.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
v.launch(model)
"""

@draccus.wrap()
def teleoperate_sim(cfg: TeleoperateSimConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        _init_rerun(session_name="teleoperation_sim")

    # 从cfg里的配置加载模型
    model = mujoco.MjModel.from_xml_path(cfg.mjcf_path)
    data = mujoco.MjData(model)

    # 初始化 MuJoCo 渲染上下文（用于录制）
    global scene, context
    scene = mujoco.MjvScene(model, maxgeom=1000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)

    # 创建相机对象
    camera_names = ["top_cam", "front_cam", "side_cam"]
    cameras = {}
    for cam_name in camera_names:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id >= 0:
            cameras[cam_name] = mujoco.MjvCamera()
            cameras[cam_name].fixedcamid = cam_id
            cameras[cam_name].type = mujoco.mjtCamera.mjCAMERA_FIXED
        else:
            print(f"警告: 相机 '{cam_name}' 未在模型中找到")

    # 初始化异步录制管理器
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    recording_manager = AsyncRecordingManager(
        model=model,
        data=data,
        cameras=cameras,
        joint_names=joint_names,
        camera_names=camera_names,
        output_dir="teleop_recordings"
    )

    # 录制控制变量
    recording_episode_id = 0
    recording_active = False

    # 键盘监听线程用于控制录制
    def keyboard_listener():
        nonlocal recording_active, recording_episode_id
        print("录制控制: 按 'r' 开始录制, 's' 停止录制, 'q' 退出")
        while True:
            try:
                key = input().strip().lower()
                if key == 'r':
                    if not recording_active:
                        recording_manager.start_recording(recording_episode_id)
                        recording_active = True
                        print(f"开始录制 Episode {recording_episode_id}")
                elif key == 's':
                    if recording_active:
                        recording_manager.stop_recording()
                        recording_active = False
                        recording_episode_id += 1
                        print(f"停止录制，下一集: {recording_episode_id}")
                elif key == 'q':
                    if recording_active:
                        recording_manager.stop_recording()
                    break
            except EOFError:
                break

    # 启动键盘监听线程
    listener_thread = threading.Thread(target=keyboard_listener, daemon=True)
    listener_thread.start()

    # Map Mujoco joint names ("1", "2", ..., "6") to indices
    mujoco_joint_names = [model.joint(i).name for i in range(model.njnt)]
    print("Mujoco joint names:", mujoco_joint_names)

    # 遥操作器的前6个关节按顺序对应MuJoCo模型的6个关节
    mujoco_indices = list(range(len(mujoco_joint_names)))
    # print("Using MuJoCo joint indices:", mujoco_indices)
    # print("Using MuJoCo joint names:", [mujoco_joint_names[i] for i in mujoco_indices])

    teleop = make_teleoperator_from_config(cfg.teleop)
    teleop.connect()

    # 计算每个控制周期需要的物理子步数，确保仿真时间跟上实时
    substeps = max(1, int(1.0 / (cfg.fps * model.opt.timestep)))

    # mujoco渲染主循环
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():

                # 获取遥操作器的关节角度指令
                action = teleop.get_action()

                # 明确按名称获取每个关节的值
                joint_values = [
                    action["shoulder_pan.pos"],
                    action["shoulder_lift.pos"],
                    action["elbow_flex.pos"],
                    action["wrist_flex.pos"],
                    action["wrist_roll.pos"],
                    action["gripper.pos"],
                ]
                # 仅当遥操作器输出为度时才做转换；否则直接按弧度使用
                # if getattr(cfg.teleop, "use_degrees", False):
                joint_values = np.deg2rad(joint_values)
                # else:
                #     joint_values = np.asarray(joint_values, dtype=float)

                # 添加关节偏置量（弧度）
                teleop_offset(joint_values)

                # 用执行器控制（而非直接写qpos），提高抓取与交互的稳定性
                for idx, val in zip(range(min(6, model.nu)), joint_values, strict=False):
                    data.ctrl[idx] = val

                # 在一个控制周期内执行多个物理子步
                for _ in range(substeps):
                    mujoco.mj_step(model, data)

                viewer.sync()

                # 打印到rerun
                if cfg.display_data:
                    for i, val in enumerate(joint_values, 1):
                        rr.log(f"action_{i}", rr.Scalar(float(val)))

                # 显示录制状态
                status = "录制中" if recording_active else "未录制"
                print(f"\r遥控状态: {status} | 仿真时间: {data.time:.2f}s", end="", flush=True)

                # 控制周期节拍（无需额外sleep，substeps已对应实时）

    except KeyboardInterrupt:
        pass
    finally:
        # 停止录制
        if recording_active:
            recording_manager.stop_recording()
        
        # 创建 metadata（如果有录制的 episodes）
        if recording_episode_id > 0:
            # 这里需要收集所有录制 episodes 的长度
            # 简单起见，我们可以假设所有 episodes 都有相同的长度或从录制管理器获取
            episode_lengths = [100] * recording_episode_id  # 临时值，需要改进
            recording_manager.create_metadata(episode_lengths)
            print(f"录制完成，共 {recording_episode_id} 个 episodes")
        
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()


if __name__ == "__main__":
    teleoperate_sim()
