# SoSEnv - System of Systems Environment

## 环境构建完成总结

完整重构了 `SoS_env.py`，将原有的三个分散模块整合为单一的、解耦的环境文件。

---

## 文件结构

### SoS_env.py (新统一文件)
```
├─ Global Constants
│  ├─ TaskNum = 5
│  ├─ DISTCON = 2.0
│  ├─ CONST1 = 0.96
│  └─ device
│
├─ SoSDataset 类
│  ├─ __init__()       # 生成 num_samples 个问题实例
│  ├─ __len__()
│  └─ __getitem__()    # 返回 (static, dynamic, [])
│
└─ SoSEnv 类 (Gym兼容)
   ├─ 公开接口
   │  ├─ reset()       # 重置环境，随机采样样本
   │  ├─ step(action)  # 执行动作
   │  ├─ render()      # 可视化（占位）
   │  └─ close()       # 清理资源
   │
   ├─ 内部状态管理
   │  ├─ _initialize_state()
   │  ├─ _get_observation()
   │  └─ _get_info()
   │
   ├─ 任务规则逻辑
   │  ├─ _apply_distance_constraint()
   │  ├─ _execute_action()
   │  ├─ _check_task_completion()
   │  ├─ _update_mask()
   │  └─ _compute_episode_reward()
   │
   └─ 终止判断
      └─ _check_termination()
```

---

## 核心特性

### 1. Gym 标准兼容
- 实现标准接口：`reset()`, `step()`, `render()`, `close()`
- 定义 `action_space` 和 `observation_space`
- 返回标准格式：`(observation, reward, terminated, truncated, info)`

### 2. 数据集集成
环境内部持有 `SoSDataset` 实例
```python
env = SoSEnv(num_nodes=50, num_samples=1000, seed=42)
obs, info = env.reset()  # ← 环境自动采样新样本
```

### 3. 完整的状态管理
```
观察空间：
├─ static: (13, num_nodes)         # 静态特征
├─ dynamic: (1, num_nodes)         # 动态特征
├─ mask: (num_nodes,)              # 合法动作掩码
├─ t: (4, TaskNum)                 # 任务剩余需求
└─ step_count: int                 # 当前步数
```

### 4. 任务规则逻辑
- **距离约束初始化**：基于系统位置距离筛选
- **掩码更新**：选择节点后自动更新可选范围
- **需求递减**：选择时根据衰减系数更新需求
- **任务完成检测**：自动判断任务是否完成
- **奖励计算**：支持单步和回合级别奖励

---

## 使用示例

### 基础用法
```python
from SoS_env import SoSEnv
import numpy as np

# 创建环境
env = SoSEnv(num_nodes=25, num_samples=100, capnum=5, seed=42)

# 单个回合
obs, info = env.reset()
done = False
step_count = 0

while not done and step_count < 20:
    # 选择合法动作
    valid_actions = np.where(obs['mask'] > 0)[0]
    action = np.random.choice(valid_actions)
    
    # 执行动作
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    step_count += 1

# 获取最终奖励
final_reward = env._compute_episode_reward()
print(f"Episode reward: {final_reward:.4f}")
```

### 与 Agent 的集成
```python
from SoS_env import SoSEnv
from model import DRL4SoS

env = SoSEnv(num_nodes=50, num_samples=1000)
actor = DRL4SoS(...)

for episode in range(num_episodes):
    obs, info = env.reset()
    
    for step in range(max_steps):
        # Agent 决策（只看 obs）
        action_probs = actor(obs)
        action = sample_action(action_probs)
        
        # 环境执行（agent 完全不知道细节）
        obs, reward, done, truncated, info = env.step(action)
        
        if done or truncated:
            break
    
    final_reward = env._compute_episode_reward()
    # 更新网络...
```

---

## 参数说明

### SoSEnv 初始化参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `num_nodes` | 50 | 系统组合数 |
| `num_samples` | 1000 | 生成的问题实例数 |
| `capnum` | 5 | 任务数量 |
| `seed` | None | 随机种子 |
| `task_num` | 5 | 任务数（通常=5） |
| `distcon` | 2.0 | 距离约束阈值 |
| `const1` | 0.96 | 选择衰减系数 |
| `max_steps` | num_nodes | 每个回合最大步数 |

### 观察空间说明

| 字段 | 形状 | 说明 |
|------|------|------|
| `static` | (13, num_nodes) | 静态特征（特征 9 为贡献值） |
| `dynamic` | (1, num_nodes) | 动态特征（通常全 0） |
| `mask` | (num_nodes,) | 1=可选, 0=不可选 |
| `t` | (4, TaskNum) | 每个任务的 4 维需求向量 |
| `step_count` | int | 当前步数 |

---

## 状态转移逻辑

```
reset() 
  ├─ 从 dataset 随机采样样本
  ├─ 初始化 t (取 static[:4, :])
  ├─ 初始化 mask = ones
  ├─ 应用距离约束
  └─ 返回 observation

step(action)
  ├─ 验证 action 合法性
  ├─ 更新 t[task_id] -= static[4+j, action] * (const1 ^ count)
  ├─ 检查任务完成 → 更新 mask
  ├─ 检查终止条件
  └─ 返回 (obs, reward, terminated, truncated, info)

_compute_episode_reward() [在回合结束时调用]
  ├─ 统计每个任务的选择次数
  ├─ 转换为配对奖励：count * (count-1) / 2
  ├─ 计算值奖励：sum(selected_values) / 20
  └─ 最终 = value_reward + pair_reward / 200
```

---

## 关键改进点

✅ **单一职责**：环境只做状态管理，网络只做决策  
✅ **Gym 兼容**：可与任何 Gym 兼容的强化学习框架集成  
✅ **数据独立**：数据集完全由环境内部管理  
✅ **解耦设计**：trainer 不需要了解数据细节  
✅ **可复现性**：每个 reset() 产生新的独立回合  

---

## 测试验证

运行 `test_env.py` 验证环境功能：
```bash
python test_env.py
```

测试覆盖：
- ✓ 环境创建
- ✓ 重置和采样
- ✓ 随机动作执行
- ✓ 状态转移
- ✓ 奖励计算
