"""
rerun-sdk 适配到老版本的 0.22.1
"""
import sys
import time
import os
import subprocess
import mujoco
import mujoco.viewer
import cv2
import glfw
import numpy as np
import pandas as pd
import json
from pathlib import Path
import threading
import rerun as rr
import shutil  # 新增导入

# ============ 配置参数 ============
# 模型和场景文件
MODEL_XML_PATH = 'myscene.xml'

# 关节名称
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# 相机名称
CAMERA_NAMES = ["top_cam", "front_cam", "side_cam"]

# 输出目录（GR00T格式）data | meta | videos
output_dir = "lerobot_dataset"

# 临时帧目录
temp_frames_dir = output_dir + "/tmp_frames"

# rerun配置
RERUN_APP_NAME = "MuJoCo Lerobot SO-101"
RERUN_DATA_FILE = "mujoco_lerobot_101_rerun.rrd"
rrd_dir = output_dir + "/rerun/" + RERUN_DATA_FILE

# 数据采集配置
NUM_EPISODES = 1  # 采集episode数量
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# 数据集分块配置
CHUNK_INDEX = 1  # 当前chunk索引
CHUNK_ID = f"{CHUNK_INDEX:03d}"  # chunk ID，如"000"

# 仿真配置
SIM_SPEED = 1.0

# 是否自动清理缓存帧目录
CLEAR_TMP_FRAMES = True

# ============ 初始化 ============

# 初始化 GLFW（隐藏窗口以使用离屏渲染）
glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(1200, 900, "mujoco", None, None)
glfw.make_context_current(window)

model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
data = mujoco.MjData(model)

# 创建固定相机对象字典
cameras = {}
for CAMERA_NAME in CAMERA_NAMES:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if cam_id < 0:
        raise ValueError(f"相机 '{CAMERA_NAME}' 未在模型中找到！")
    cameras[CAMERA_NAME] = mujoco.MjvCamera()
    cameras[CAMERA_NAME].fixedcamid = cam_id
    cameras[CAMERA_NAME].type = mujoco.mjtCamera.mjCAMERA_FIXED

# 创建渲染场景和上下文
scene = mujoco.MjvScene(model, maxgeom=1000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)

# 数据采集缓冲区
class DataCollector:
    def __init__(self, episode_id=0):
        self.episode_id = episode_id
        self.timestamps = []
        self.frame_indices = []
        self.joint_positions = []
        self.joint_velocities = []
        self.joint_torques = []  # 如果有控制信号
        self.gripper_actions = []  # 如果有夹爪
        
        # 用于保存带时间戳的图片文件名
        self.frame_filenames = {}  # camera_name -> [filename1, filename2, ...]
        for cam_name in CAMERA_NAMES:
            self.frame_filenames[cam_name] = []
    
    def add_step(self, sim_time, frame_idx, joint_pos, joint_vel, joint_torque=None, gripper=None):
        """添加一个时间步的数据"""
        self.timestamps.append(sim_time)
        self.frame_indices.append(frame_idx)
        self.joint_positions.append(joint_pos.copy())
        self.joint_velocities.append(joint_vel.copy())
        
        if joint_torque is not None:
            self.joint_torques.append(joint_torque.copy())
        else:
            # 如果没有力矩数据，用零填充
            self.joint_torques.append(np.zeros_like(joint_pos))
            
        if gripper is not None:
            self.gripper_actions.append(gripper)
        else:
            self.gripper_actions.append(0.0)  # 默认夹爪状态
    
    def add_frame_filename(self, cam_name, filename):
        """记录对应时间戳的图片文件名"""
        self.frame_filenames[cam_name].append(filename)
    
    def to_parquet(self, output_path):
        """转换为GR00T需要的parquet格式"""
        data_dict = {
            'timestamp': self.timestamps,
            'frame_index': self.frame_indices,
            'episode_index': [self.episode_id] * len(self.timestamps),
            
            # 状态数据（观测）- 实际的关节角度和速度
            'state.joint_positions': self.joint_positions,
            'state.joint_velocities': self.joint_velocities,
            'state.gripper': [[g] for g in self.gripper_actions],  # 转为列表格式
            
            # 动作数据（控制指令）- 实际的电机力矩/控制信号
            'action.joint_torques': self.joint_torques,  # 电机实时力矩控制信号
            'action.gripper': [[g] for g in self.gripper_actions],
            
            # 任务描述
            'task_index': [0] * len(self.timestamps),  # 所有数据属于同一任务
        }
        
        df = pd.DataFrame(data_dict)
        df.to_parquet(output_path, index=False)
        return len(self.timestamps)

def get_image(camera, w, h):
    """从指定相机渲染一帧，返回 BGR numpy 图像"""
    viewport = mujoco.MjrRect(0, 0, w, h)
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.zeros((h, w), dtype=np.float64)
    mujoco.mjr_readPixels(bgr, depth, viewport, context)
    return np.flipud(bgr)  # 返回BGR

def prepare_frame_dirs(cameras, episode_id):
    """为每个相机创建带时间戳的图片序列目录"""
    base = os.path.join(temp_frames_dir, f"episode_{episode_id:06d}")
    os.makedirs(base, exist_ok=True)
    dirs = {}
    for name in cameras:
        d = os.path.join(base, name)
        os.makedirs(d, exist_ok=True)
        dirs[name] = d
    return dirs

def save_frame_with_timestamp(frame_dirs, images, sim_time, frame_idx, data_collector):
    """使用仿真时间戳保存图片"""
    # 使用仿真时间作为文件名，确保时间戳精确
    timestamp_str = f"{sim_time:.6f}".replace('.', '_')  # 避免文件名中的点号
    
    for cam_name, img in images.items():
        # 文件名格式：timestamp_{仿真时间}_frame_{帧索引}.png
        filename = f"timestamp_{timestamp_str}_frame_{frame_idx:06d}.png"
        path = os.path.join(frame_dirs[cam_name], filename)
        cv2.imwrite(path, img)
        
        # 记录文件名到数据收集器
        data_collector.add_frame_filename(cam_name, filename)

def create_video_from_timestamped_frames(frame_dirs, out_dir, data_collector, episode_id):
    """从带时间戳的图片创建视频，并生成timestamp映射"""
    os.makedirs(out_dir, exist_ok=True)
    
    for cam_name, frame_dir in frame_dirs.items():
        # 为每个相机创建子文件夹
        cam_video_dir = os.path.join(out_dir, cam_name)
        os.makedirs(cam_video_dir, exist_ok=True)
        
        # 按时间戳排序文件
        frame_files = data_collector.frame_filenames[cam_name]
        timestamps = data_collector.timestamps
        
        # 创建临时的按顺序命名的符号链接
        temp_ordered_dir = os.path.join(frame_dir, "ordered")
        os.makedirs(temp_ordered_dir, exist_ok=True)
        
        for i, (timestamp, filename) in enumerate(zip(timestamps, frame_files)):
            src = os.path.join(frame_dir, filename)
            dst = os.path.join(temp_ordered_dir, f"frame_{i:06d}.png")
            if os.path.exists(src):
                os.symlink(os.path.abspath(src), dst)
        
        # 计算平均帧率用于ffmpeg
        if len(timestamps) > 1:
            total_time = timestamps[-1] - timestamps[0]
            avg_fps = len(timestamps) / total_time if total_time > 0 else 30.0
        else:
            avg_fps = 30.0
            
        # 使用ffmpeg编码
        in_pattern = os.path.join(temp_ordered_dir, "frame_%06d.png")
        out_file = os.path.join(cam_video_dir, f"episode_{episode_id:06d}.mp4")
        # out_file = cam_video_dir + f"/episode_{episode_id:06d}.mp4"
        
        cmd = [
            "ffmpeg", "-y",
            "-framerate", f"{avg_fps:.6f}",
            "-i", in_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            out_file
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print(f"视频已生成: {out_file}，平均帧率: {avg_fps:.2f} FPS。")
        except subprocess.CalledProcessError:
            print(f"ffmpeg 编码失败: {cam_name}", file=sys.stderr)
        
        # 清理临时符号链接
        for f in os.listdir(temp_ordered_dir):
            os.unlink(os.path.join(temp_ordered_dir, f))
        os.rmdir(temp_ordered_dir)

def create_lerobot_metadata(output_dir, episode_data_list):
    """创建GR00T需要的metadata文件"""
    meta_dir = os.path.join(output_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    
    # 1. modality.json - 定义数据模态
    modality_config = {
        "state": {
            "joint_positions": {"start": 0, "end": 7},  # 实际关节角度
            "joint_velocities": {"start": 7, "end": 14}, # 实际关节速度
            "gripper": {"start": 14, "end": 15}          # 夹爪位置
        },
        "action": {
            "joint_torques": {"start": 0, "end": 7},     # 电机力矩控制信号  
            "gripper": {"start": 7, "end": 8}            # 夹爪控制信号
        },
        "video": {}
    }
    
    # 添加视频模态
    for cam_name in CAMERA_NAMES:
        modality_config["video"][cam_name] = {
            "original_key": f"observation.images.{cam_name}"
        }
    
    with open(os.path.join(meta_dir, "modality.json"), 'w') as f:
        json.dump(modality_config, f, indent=2)
    
    # 2. episodes.jsonl - 每个episode的信息
    with open(os.path.join(meta_dir, "episodes.jsonl"), 'w') as f:
        for i, length in enumerate(episode_data_list):
            episode_data = {
                "episode_index": i,
                "length": length,
                "timestamp": 0.0,
                "task_id": 0,
            }
            f.write(json.dumps(episode_data) + '\n')
    
    # 3. tasks.jsonl - 任务信息
    with open(os.path.join(meta_dir, "tasks.jsonl"), 'w') as f:
        task_data = {
            "task_index": 0,
            "task_id": "mujoco_manipulation",
            "task_description": "MuJoCo manipulation task"
        }
        f.write(json.dumps(task_data) + '\n')
    
    # 4. info.json - 数据集基本信息
    info_data = {
        "dataset_name": "mujoco_dataset",
        "fps": 30.0,  # 平均帧率
        "total_episodes": len(episode_data_list),
        "total_frames": sum(episode_data_list),
        "chunks_size": 1000,
        "data_path": f"data/chunk-{CHUNK_ID}/episode_{{episode_index:06d}}.parquet",
        "video_path": f"videos/chunk-{CHUNK_ID}/{{video_key}}/episode_{{episode_index:06d}}.mp4"
    }
    
    with open(os.path.join(meta_dir, "info.json"), 'w') as f:
        json.dump(info_data, f, indent=2)

# 主程序
def run_data_collection(episode_id=0):
    """运行单个episode的数据采集"""
    exit_flag = [False]  # 使用列表以便在嵌套函数中修改
    
    def keyboard_listener():
        while True:
            try:
                key = input()  # 等待Enter键（空输入）
                if key == '':  # Enter键
                    exit_flag[0] = True
                    break
            except EOFError:
                break
    
    # 启动键盘监听线程
    listener_thread = threading.Thread(target=keyboard_listener, daemon=True)
    listener_thread.start()
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        width, height = IMAGE_WIDTH, IMAGE_HEIGHT
        data_collector = DataCollector(episode_id)
        
        # 创建图片序列保存目录
        frame_dirs = prepare_frame_dirs(cameras, episode_id)
        
        frame_idx = 0
        fps_counter = 0
        last_fps_time = time.time()
        
        # 固定步长仿真控制
        sim_speed = SIM_SPEED
        acc = 0.0
        last_wall = time.time()
        
        while viewer.is_running() and not exit_flag[0]:
            now = time.time()
            acc += (now - last_wall) * sim_speed
            last_wall = now
            
            h = model.opt.timestep
            max_catchup = int(10 / h)
            steps = 0
            while acc >= h and steps < max_catchup:
                mujoco.mj_step(model, data)
                acc -= h
                steps += 1
            
            # 获取当前仿真时间和关节数据
            sim_time = data.time
            # 使用 rerun 的 set_time_seconds 接口来设置当前时间戳
            rr.set_time_seconds("timestamp", sim_time)  # 设置rerun时间
            
            # 根据名称获取关节索引
            joint_indices = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]

            # 获取关节位置、速度和控制信号
            joint_pos = np.array([data.qpos[idx] for idx in joint_indices])
            joint_vel = np.array([data.qvel[idx] for idx in joint_indices])
            
            # 获取电机控制信号（实时力矩）
            joint_torque = np.array([data.ctrl[idx] for idx in joint_indices]).copy() if data.ctrl is not None else np.zeros_like(joint_pos)  # 实际电机控制力矩
            
            # 夹爪控制直接取 joint_indices 最后一个
            # gripper_action = data.ctrl[joint_indices[-1]] if data.ctrl is not None else 0.0
            gripper_action = joint_torque[-1] if data.ctrl is not None else 0.0
            
            # 记录关节数据到rerun
            for i, name in enumerate(JOINT_NAMES):
                rr.log(f"joints/{name}/position", rr.Scalar(float(joint_pos[i])))
                rr.log(f"joints/{name}/velocity", rr.Scalar(float(joint_vel[i])))
                rr.log(f"joints/{name}/torque", rr.Scalar(float(joint_torque[i])))
            rr.log("gripper", rr.Scalar(float(gripper_action)))
            
            # 渲染三台相机(BGR)
            images = {}
            for CAMERA_NAME in CAMERA_NAMES:
                images[CAMERA_NAME] = get_image(cameras[CAMERA_NAME], width, height)

            # 记录图像到rerun(BGR)
            for cam_name, img in images.items():
                rr.log(f"cameras/{cam_name}", rr.Image(img))
            
            # 将images字典中的所有图片从BGR转为RGB
            for cam_name in images:
                images[cam_name] = cv2.cvtColor(images[cam_name], cv2.COLOR_BGR2RGB)

            # 保存带时间戳的图片
            save_frame_with_timestamp(frame_dirs, images, sim_time, frame_idx, data_collector)
            
            # 记录数据到收集器
            data_collector.add_step(sim_time, frame_idx, joint_pos, joint_vel, joint_torque, gripper_action)
            
            frame_idx += 1
            
            # 显示Mujoco窗口
            viewer.sync()
            
            fps_counter += 1
            if time.time() - last_fps_time >= 1.0:
                print(f"\r实时的 Viewer FPS: {fps_counter}", end="", flush=True)
                fps_counter = 0
                last_fps_time = time.time()
        
        print(f"\n完成Episode {episode_id}，总帧数: {frame_idx}张图片。")
        
        # 保存parquet数据
        data_dir = os.path.join(output_dir, "data", f"chunk-{CHUNK_ID}")
        os.makedirs(data_dir, exist_ok=True)
        parquet_path = os.path.join(data_dir, f"episode_{episode_id:06d}.parquet")
        episode_length = data_collector.to_parquet(parquet_path)
        
        # 生成视频
        video_dir = os.path.join(output_dir, "videos", f"chunk-{CHUNK_ID}")
        os.makedirs(video_dir, exist_ok=True)
        create_video_from_timestamped_frames(frame_dirs, video_dir, data_collector, episode_id)

        # 统计并打印录制视频时长
        if frame_idx > 1:
            duration = data_collector.timestamps[-1] - data_collector.timestamps[0]
            print(f"录制视频时长: {duration:.3f} 秒")
        else:
            print("录制视频时长: 0 秒")
        
        cv2.destroyAllWindows()
        return episode_length

# 主程序
if __name__ == "__main__":
    # 必须清理缓存帧目录，否则导致衔接到最长的视频
    if os.path.exists(temp_frames_dir):
        shutil.rmtree(temp_frames_dir)
    
    # 初始化rerun
    rr.init(RERUN_APP_NAME, spawn=True)
    
    # 采集多个episodes
    episode_lengths = []
    num_episodes = NUM_EPISODES  # 采集episodes数量
    for ep_id in range(num_episodes):
        print(f"开始采集Episode {ep_id}")
        # 使用 rerun 的 set_time_sequence 接口来设置 episode 序列
        rr.set_time_sequence("episode", ep_id)  # 设置episode时间序列
        length = run_data_collection(ep_id)
        episode_lengths.append(length)
    
    # 创建metadata文件
    create_lerobot_metadata(output_dir, episode_lengths)
    
    # 保存rerun数据
    os.makedirs(os.path.dirname(rrd_dir), exist_ok=True)
    rr.save(rrd_dir)
    
    print(f"数据采集完成！GR00T格式数据保存在: {os.path.abspath(output_dir)}")
    print(f"rerun数据保存为: {rrd_dir}")

    # 自动清理缓存帧目录
    if CLEAR_TMP_FRAMES and os.path.exists(temp_frames_dir):
        shutil.rmtree(temp_frames_dir)

    print("采集流程结束。")