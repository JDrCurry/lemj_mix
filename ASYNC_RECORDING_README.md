# 异步遥控录制集成说明

## 概述

通过引用方式将 `rr_tele_record_lemj.py` 的录制功能异步集成到 `teleoperate_sim.py` 中，实现遥控过程中同时进行视频和数据录制。

## 新增文件

- `async_recording.py`: 异步录制管理器，从 `rr_tele_record_lemj.py` 提取的核心录制功能

## 修改的文件

- `teleoperate_sim.py`: 添加异步录制支持

## 使用方法

### 启动遥控

```bash
python teleoperate_sim.py --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=my_leader --display_data True
```

### 录制控制

程序启动后，会显示录制控制提示：

```
录制控制: 按 'r' 开始录制, 's' 停止录制, 'q' 退出
```

- **r**: 开始录制当前 episode
- **s**: 停止当前录制，开始下一个 episode
- **q**: 退出程序（会自动停止录制）

### 输出文件

录制的数据保存在 `teleop_recordings/` 目录下，包含：

- `data/chunk-001/episode_XXXXXX.parquet`: 关节数据
- `videos/chunk-001/{camera}/episode_XXXXXX.mp4`: 视频文件
- `meta/`: 元数据文件（modality.json, episodes.jsonl, tasks.jsonl, info.json）
- `rerun/`: rerun 可视化数据（如果启用）

## 技术实现

### 异步录制

- 使用 `threading.Thread` 在后台运行录制循环
- 录制线程定期收集仿真状态、渲染图像、保存数据
- 主线程继续处理遥控输入，不受录制影响

### 共享资源

- MuJoCo `model` 和 `data` 对象在主线程和录制线程间共享
- 使用全局 `scene` 和 `context` 进行渲染
- 相机对象在初始化时创建

### 数据格式

- 遵循 GR00T 数据集格式
- 支持多摄像头录制
- 包含关节位置、速度、力矩和图像数据

## 注意事项

1. **依赖**: 需要安装 `rerun-sdk==0.22.1` 和相关依赖
2. **相机**: 确保模型中包含 `top_cam`, `front_cam`, `side_cam` 相机
3. **性能**: 录制会消耗额外 CPU 资源，但通过异步处理最小化影响
4. **存储**: 确保有足够磁盘空间存储视频和数据文件

## 故障排除

- 如果录制失败，检查相机名称是否正确
- 如果视频生成失败，确保安装了 `ffmpeg`
- 如果 rerun 相关错误，检查 rerun 版本是否为 0.22.1