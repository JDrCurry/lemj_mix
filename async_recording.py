"""
异步录制模块 - 从 rr_tele_record_lemj.py 提取的核心录制功能
用于在 teleoperate_sim.py 中异步进行数据和视频录制
"""
import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import json
import threading
import rerun as rr
from pathlib import Path
import shutil
import subprocess

class AsyncRecordingManager:
    """异步录制管理器"""
    
    def __init__(self, model, data, cameras, joint_names, camera_names, output_dir="lerobot_dataset"):
        self.model = model
        self.data = data
        self.cameras = cameras
        self.joint_names = joint_names
        self.camera_names = camera_names
        self.output_dir = output_dir
        
        # 录制配置
        self.image_width = 640
        self.image_height = 480
        self.chunk_index = 1
        self.chunk_id = f"{self.chunk_index:03d}"
        
        # 录制状态
        self.is_recording = False
        self.recording_thread = None
        self.stop_event = threading.Event()
        self.current_episode_id = 0
        
        # 数据缓冲
        self.data_collector = None
        self.frame_dirs = None
        
        # 临时目录
        self.temp_frames_dir = os.path.join(output_dir, "tmp_frames")
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
    
    def start_recording(self, episode_id=0):
        """开始异步录制"""
        if self.is_recording:
            print("录制已在进行中")
            return
        
        self.is_recording = True
        self.current_episode_id = episode_id
        self.stop_event.clear()
        
        # 初始化数据收集器和帧目录
        self.data_collector = DataCollector(episode_id)
        self.frame_dirs = self._prepare_frame_dirs(episode_id)
        
        # 启动录制线程
        self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.recording_thread.start()
        print(f"开始异步录制 Episode {episode_id}")
    
    def stop_recording(self):
        """停止录制"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.stop_event.set()
        
        # 等待线程结束
        if self.recording_thread:
            self.recording_thread.join(timeout=5.0)
        
        # 处理录制的数据
        self._finalize_recording()
        print(f"停止录制 Episode {self.current_episode_id}")
    
    def _recording_loop(self):
        """录制主循环"""
        frame_idx = 0
        fps_counter = 0
        last_fps_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                # 获取当前仿真时间
                sim_time = self.data.time
                
                # 设置 rerun 时间
                rr.set_time_seconds("timestamp", sim_time)
                
                # 获取关节数据
                joint_indices = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                               for name in self.joint_names]
                
                joint_pos = np.array([self.data.qpos[idx] for idx in joint_indices])
                joint_vel = np.array([self.data.qvel[idx] for idx in joint_indices])
                joint_torque = np.array([self.data.ctrl[idx] for idx in joint_indices]) if self.data.ctrl is not None else np.zeros_like(joint_pos)
                gripper_action = joint_torque[-1] if self.data.ctrl is not None else 0.0
                
                # 记录到 rerun
                for i, name in enumerate(self.joint_names):
                    rr.log(f"joints/{name}/position", rr.Scalar(float(joint_pos[i])))
                    rr.log(f"joints/{name}/velocity", rr.Scalar(float(joint_vel[i])))
                    rr.log(f"joints/{name}/torque", rr.Scalar(float(joint_torque[i])))
                rr.log("gripper", rr.Scalar(float(gripper_action)))
                
                # 渲染图像
                images = {}
                for cam_name in self.camera_names:
                    images[cam_name] = self._get_image(self.cameras[cam_name], self.image_width, self.image_height)
                
                # 记录图像到 rerun
                for cam_name, img in images.items():
                    rr.log(f"cameras/{cam_name}", rr.Image(img))
                
                # 转换颜色空间
                for cam_name in images:
                    images[cam_name] = cv2.cvtColor(images[cam_name], cv2.COLOR_BGR2RGB)
                
                # 保存帧
                self._save_frame_with_timestamp(images, sim_time, frame_idx)
                
                # 添加到数据收集器
                self.data_collector.add_step(sim_time, frame_idx, joint_pos, joint_vel, joint_torque, gripper_action)
                
                frame_idx += 1
                
                # FPS 统计
                fps_counter += 1
                if time.time() - last_fps_time >= 1.0:
                    print(f"\r录制 FPS: {fps_counter}", end="", flush=True)
                    fps_counter = 0
                    last_fps_time = time.time()
                
                # 短暂休眠避免占用过多CPU
                time.sleep(0.01)
                
            except Exception as e:
                print(f"录制错误: {e}")
                break
        
        print(f"\n录制完成，帧数: {frame_idx}")
    
    def _prepare_frame_dirs(self, episode_id):
        """准备帧目录"""
        base = os.path.join(self.temp_frames_dir, f"episode_{episode_id:06d}")
        os.makedirs(base, exist_ok=True)
        dirs = {}
        for name in self.camera_names:
            d = os.path.join(base, name)
            os.makedirs(d, exist_ok=True)
            dirs[name] = d
        return dirs
    
    def _get_image(self, camera, w, h):
        """渲染图像"""
        # 注意：这里需要访问全局的 scene 和 context
        # 在实际使用时需要确保这些在主线程中初始化
        global scene, context
        viewport = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjv_updateScene(self.model, self.data, mujoco.MjvOption(), None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)
        bgr = np.zeros((h, w, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(bgr, None, viewport, context)
        return np.flipud(bgr)
    
    def _save_frame_with_timestamp(self, images, sim_time, frame_idx):
        """保存带时间戳的帧"""
        timestamp_str = f"{sim_time:.6f}".replace('.', '_')
        
        for cam_name, img in images.items():
            filename = f"timestamp_{timestamp_str}_frame_{frame_idx:06d}.png"
            path = os.path.join(self.frame_dirs[cam_name], filename)
            cv2.imwrite(path, img)
            self.data_collector.add_frame_filename(cam_name, filename)
    
    def _finalize_recording(self):
        """完成录制处理"""
        if not self.data_collector:
            return
        
        episode_id = self.current_episode_id
        
        # 保存 parquet
        data_dir = os.path.join(self.output_dir, "data", f"chunk-{self.chunk_id}")
        os.makedirs(data_dir, exist_ok=True)
        parquet_path = os.path.join(data_dir, f"episode_{episode_id:06d}.parquet")
        episode_length = self.data_collector.to_parquet(parquet_path)
        
        # 生成视频
        video_dir = os.path.join(self.output_dir, "videos", f"chunk-{self.chunk_id}")
        os.makedirs(video_dir, exist_ok=True)
        self._create_video_from_frames(video_dir, episode_id)
        
        # 打印统计
        if len(self.data_collector.timestamps) > 1:
            duration = self.data_collector.timestamps[-1] - self.data_collector.timestamps[0]
            print(f"录制时长: {duration:.3f} 秒")
    
    def _create_video_from_frames(self, video_dir, episode_id):
        """从帧创建视频"""
        for cam_name in self.camera_names:
            cam_video_dir = os.path.join(video_dir, cam_name)
            os.makedirs(cam_video_dir, exist_ok=True)
            
            frame_files = self.data_collector.frame_filenames[cam_name]
            timestamps = self.data_collector.timestamps
            
            if not frame_files:
                continue
            
            # 创建有序的临时链接
            temp_ordered_dir = os.path.join(self.frame_dirs[cam_name], "ordered")
            os.makedirs(temp_ordered_dir, exist_ok=True)
            
            for i, (timestamp, filename) in enumerate(zip(timestamps, frame_files)):
                src = os.path.join(self.frame_dirs[cam_name], filename)
                dst = os.path.join(temp_ordered_dir, f"frame_{i:06d}.png")
                if os.path.exists(src):
                    os.symlink(os.path.abspath(src), dst)
            
            # 计算帧率
            if len(timestamps) > 1:
                total_time = timestamps[-1] - timestamps[0]
                avg_fps = len(timestamps) / total_time if total_time > 0 else 30.0
            else:
                avg_fps = 30.0
            
            # ffmpeg 编码
            in_pattern = os.path.join(temp_ordered_dir, "frame_%06d.png")
            out_file = os.path.join(cam_video_dir, f"episode_{episode_id:06d}.mp4")
            
            cmd = [
                "ffmpeg", "-y", "-framerate", f"{avg_fps:.6f}",
                "-i", in_pattern, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "fast", out_file
            ]
            
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                print(f"视频生成: {out_file}")
            except subprocess.CalledProcessError:
                print(f"ffmpeg 失败: {cam_name}")
            
    def create_metadata(self, episode_lengths):
        """创建 GR00T metadata 文件"""
        meta_dir = os.path.join(self.output_dir, "meta")
        os.makedirs(meta_dir, exist_ok=True)
        
        # modality.json
        modality_config = {
            "state": {
                "joint_positions": {"start": 0, "end": 7},
                "joint_velocities": {"start": 7, "end": 14},
                "gripper": {"start": 14, "end": 15}
            },
            "action": {
                "joint_torques": {"start": 0, "end": 7},
                "gripper": {"start": 7, "end": 8}
            },
            "video": {}
        }
        
        for cam_name in self.camera_names:
            modality_config["video"][cam_name] = {
                "original_key": f"observation.images.{cam_name}"
            }
        
        with open(os.path.join(meta_dir, "modality.json"), 'w') as f:
            json.dump(modality_config, f, indent=2)
        
        # episodes.jsonl
        with open(os.path.join(meta_dir, "episodes.jsonl"), 'w') as f:
            for i, length in enumerate(episode_lengths):
                episode_data = {
                    "episode_index": i,
                    "length": length,
                    "timestamp": 0.0,
                    "task_id": 0,
                }
                f.write(json.dumps(episode_data) + '\n')
        
        # tasks.jsonl
        with open(os.path.join(meta_dir, "tasks.jsonl"), 'w') as f:
            task_data = {
                "task_index": 0,
                "task_id": "teleoperation_task",
                "task_description": "Teleoperation task with video recording"
            }
            f.write(json.dumps(task_data) + '\n')
        
        # info.json
        info_data = {
            "dataset_name": "teleoperation_dataset",
            "fps": 30.0,
            "total_episodes": len(episode_lengths),
            "total_frames": sum(episode_lengths),
            "chunks_size": 1000,
            "data_path": f"data/chunk-{self.chunk_id}/episode_{{episode_index:06d}}.parquet",
            "video_path": f"videos/chunk-{self.chunk_id}/{{video_key}}/episode_{{episode_index:06d}}.mp4"
        }
        
        with open(os.path.join(meta_dir, "info.json"), 'w') as f:
            json.dump(info_data, f, indent=2)


class DataCollector:
    """数据收集器"""
    def __init__(self, episode_id=0):
        self.episode_id = episode_id
        self.timestamps = []
        self.frame_indices = []
        self.joint_positions = []
        self.joint_velocities = []
        self.joint_torques = []
        self.gripper_actions = []
        self.frame_filenames = {cam: [] for cam in ["top_cam", "front_cam", "side_cam"]}
    
    def add_step(self, sim_time, frame_idx, joint_pos, joint_vel, joint_torque=None, gripper=None):
        self.timestamps.append(sim_time)
        self.frame_indices.append(frame_idx)
        self.joint_positions.append(joint_pos.copy())
        self.joint_velocities.append(joint_vel.copy())
        
        if joint_torque is not None:
            self.joint_torques.append(joint_torque.copy())
        else:
            self.joint_torques.append(np.zeros_like(joint_pos))
            
        if gripper is not None:
            self.gripper_actions.append(gripper)
        else:
            self.gripper_actions.append(0.0)
    
    def add_frame_filename(self, cam_name, filename):
        self.frame_filenames[cam_name].append(filename)
    
    def to_parquet(self, output_path):
        data_dict = {
            'timestamp': self.timestamps,
            'frame_index': self.frame_indices,
            'episode_index': [self.episode_id] * len(self.timestamps),
            'state.joint_positions': self.joint_positions,
            'state.joint_velocities': self.joint_velocities,
            'state.gripper': [[g] for g in self.gripper_actions],
            'action.joint_torques': self.joint_torques,
            'action.gripper': [[g] for g in self.gripper_actions],
            'task_index': [0] * len(self.timestamps),
        }
        
        df = pd.DataFrame(data_dict)
        df.to_parquet(output_path, index=False)
        return len(self.timestamps)


# 导入 mujoco 相关
import mujoco