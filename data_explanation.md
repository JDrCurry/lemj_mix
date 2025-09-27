# MuJoCo -> GR00T 数据映射说明

## 正确的数据含义：

### 状态数据（State - 观测值）
- `state.joint_positions`: data.qpos - **实际的关节角度**（传感器读数）
- `state.joint_velocities`: data.qvel - **实际的关节速度**（传感器读数）
- `state.gripper`: 夹爪的实际位置/开合程度

### 动作数据（Action - 控制指令）
- `action.joint_torques`: data.ctrl - **电机的实时力矩控制信号**（您发给电机的指令）
- `action.gripper`: 夹爪的控制指令

## 关键区别：
```python
# 错误理解：
action = 下一时刻的期望关节角度  # 这不是控制信号！

# 正确理解：
action = 当前时刻发给电机的力矩指令  # 这才是真正的控制信号！
```

## 实际的控制流程：
```
1. 观测：读取传感器 -> state.joint_positions, state.joint_velocities
2. 决策：AI模型计算 -> 输出 action.joint_torques（力矩指令）
3. 执行：电机接收力矩指令 -> 产生运动 -> 改变关节角度
4. 反馈：新的关节角度被传感器读取 -> 下一步的state
```

## 在您的MuJoCo代码中：
- `data.qpos`：关节的实际角度（状态观测）
- `data.qvel`：关节的实际速度（状态观测）  
- `data.ctrl`：发送给执行器的控制信号（动作指令）

这样GR00T学习的就是：
**"看到当前关节角度和速度 -> 决定发送什么力矩给电机"**

而不是：
**"看到当前关节角度 -> 决定下一时刻的目标角度"**（这是错误的）

## 如果您使用位置控制而非力矩控制：
```python
# 如果您的MuJoCo模型使用位置控制器
action_joint_target_positions = data.ctrl  # 目标位置指令
state_joint_actual_positions = data.qpos   # 实际位置读数

# 那么modality.json应该是：
"action": {
    "joint_target_positions": {"start": 0, "end": 7},  # 位置控制指令
    "gripper": {"start": 7, "end": 8}
}
```