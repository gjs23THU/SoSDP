"""SoS Environment - Combines SoSDataset and SoSEnv for System of Systems optimization"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==================== Global Constants ====================
NUM_TASKS = 5  # 任务数量（常量）+1（加了固定成本）
NUM_SYSTEMS = 50  # 候选子系统数量（常量）
DISTCON = 2.0  # 距离约束阈值
CONST1 = 0.96  # 衰减系数


# ==================== SoSDataset Class ====================
class SoSDataset:
    """Generates synthetic SoS problem instances"""

    def __init__(self, num_systems=NUM_SYSTEMS, num_caps=5, num_samples=1e4, seed=None):
        """
        Args:
            num_systems: 候选子系统数量
            num_tasks: 任务数量（通常为5）
            num_samples: 生成的问题实例数量
            seed: 随机种子（用于可复现性）
        """

        if seed is None:
            seed = np.random.randint(123456789)

        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        num_samples = int(num_samples)
        
        # Generate task-specific capability requirements (T1-T5)
        T1 = rng.integers(0, 15, (num_samples, 1, 4)).astype(np.float32)
        T2 = rng.integers(0, 15, (num_samples, 1, 4)).astype(np.float32)
        T3 = rng.integers(0, 15, (num_samples, 1, 4)).astype(np.float32)
        T4 = rng.integers(0, 15, (num_samples, 1, 4)).astype(np.float32)
        T5 = rng.integers(0, 15, (num_samples, 1, 4)).astype(np.float32)
        # Generate location distributions for each task
        dist1 = rng.uniform(-1, 1, (num_samples, 1, 2)).astype(np.float32)
        dist2 = rng.uniform(-1, 1, (num_samples, 1, 2)).astype(np.float32)
        dist3 = rng.uniform(-1, 1, (num_samples, 1, 2)).astype(np.float32)
        dist4 = rng.uniform(-1, 1, (num_samples, 1, 2)).astype(np.float32)
        dist5 = rng.uniform(-1, 1, (num_samples, 1, 2)).astype(np.float32)
        distsys = rng.uniform(-1, 1, (num_samples, 2, num_systems)).astype(np.float32)

        # Generate base system attributes
        # 生成基础系统前n-1个属性，每个系统有 num_caps(5)-1个属性维度
        dataset1 = rng.integers(0, 10, (num_samples, num_caps - 1, num_systems)).astype(np.float32)
        #print(f"dataset1.shape: {dataset1.shape}")
        # 对于每个系统，计算其固定成本属性值
        temp1 = np.sum(dataset1, axis=1, keepdims=True).astype(np.float32)

        # 调整每个系统固定成本属性值，增加随机扰动
        temp1 = temp1 * (1.0 + 1.8 * rng.random((num_samples, 1, num_systems)).astype(np.float32))
        temp1 = temp1 / 10
        # dataset1: (num_samples, num_caps-1, num_systems), temp1: (num_samples, 1, num_systems),组合后得到最终的系统属性数据集
        self.dataset = np.concatenate((np.concatenate((dataset1, temp1), axis=1), distsys), axis=1).astype(np.float32)
        # dataset2: (num_samples, num_caps + num_caps - 1 + 4, num_systems * NUM_TASKS)，包含每个系统对于每个任务的属性（13维，任务需求4，系统能力5，节点坐标2，系统坐标2）
        self.dataset2 = np.zeros((num_samples, num_caps + num_caps - 1 + 4, num_systems * NUM_TASKS), dtype=np.float32)
        #print(f"self.dataset.shape: {self.dataset.shape}, self.dataset2.shape: {self.dataset2.shape}")
        # Combine all features into dataset2
        for i in range(num_samples):
            for j in range(num_systems):
                # (T1[i].squeeze(0)=任务1需求4维,self.dataset[i, :, j]=系统属性13维,dist1[i].squeeze(0)=任务1位置2维)
                self.dataset2[i, :, j * NUM_TASKS + 0] = np.concatenate((np.concatenate((T1[i, 0], self.dataset[i, :, j]), axis=0), dist1[i, 0]), axis=0)
                self.dataset2[i, :, j * NUM_TASKS + 1] = np.concatenate((np.concatenate((T2[i, 0], self.dataset[i, :, j]), axis=0), dist2[i, 0]), axis=0)
                self.dataset2[i, :, j * NUM_TASKS + 2] = np.concatenate((np.concatenate((T3[i, 0], self.dataset[i, :, j]), axis=0), dist3[i, 0]), axis=0)
                self.dataset2[i, :, j * NUM_TASKS + 3] = np.concatenate((np.concatenate((T4[i, 0], self.dataset[i, :, j]), axis=0), dist4[i, 0]), axis=0)
                self.dataset2[i, :, j * NUM_TASKS + 4] = np.concatenate((np.concatenate((T5[i, 0], self.dataset[i, :, j]), axis=0), dist5[i, 0]), axis=0)
            # 系统动态指标：0可用 1不可用，初始全为0
            self.dynamic = np.zeros((num_samples, 1, num_systems * NUM_TASKS), dtype=np.float32)
        self.num_nodes = num_systems * NUM_TASKS
        # print(f"num_nodes: {self.num_nodes}, size: {num_samples}")
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (self.dataset2[idx], self.dynamic[idx], [])


# ==================== SoSEnv Class ====================
class SoSEnv(gym.Env):
    """System of Systems Optimization Environment (Gym-compatible)"""
    def __init__(
        self,
        num_systems=NUM_SYSTEMS,
        num_samples=1000,
        num_caps=5,
        seed=None,
        distcon=DISTCON,
        const1=CONST1,
        max_steps=None,
    ):
        super().__init__()

        self.num_caps = int(num_caps)
        self.num_tasks = NUM_TASKS
        self.num_systems = int(num_systems)
        self.distcon = float(distcon)
        self.const1 = float(const1)

        self.dataset = SoSDataset(
            num_systems=self.num_systems,
            num_caps=self.num_caps,
            num_samples=num_samples,
            seed=seed,
        )

        self.num_nodes = self.dataset.num_nodes
        self.max_steps = int(max_steps) if max_steps is not None else self.num_nodes

        self.action_space = spaces.Discrete(self.num_nodes)
        self.observation_space = spaces.Dict(
            {
                # 观察空间包含静态特征：13个维度（任务需求4维 + 系统属性13维 - 1 + 任务位置2维），动态特征：1个维度（当前状态），掩码：1个维度（可选动作），任务需求：4个维度（每个任务的剩余需求），步数计数器
                "static": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(13, self.num_nodes),
                    dtype=np.float32,
                ),
                "dynamic": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1, self.num_nodes),
                    dtype=np.float32,
                ),
                "mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.num_nodes,),
                    dtype=np.float32,
                ),
                "t": spaces.Box(
                    low=0,
                    high=15,
                    shape=(4, self.num_tasks),
                    dtype=np.float32,
                ),
                "step_count": spaces.Discrete(self.max_steps + 1),
            }
        )

        self.static = None
        self.dynamic = None
        self.t = None
        self.mask = None
        self.selected_nodes = []
        self.task_counts = None
        self.episode_values = None
        self.step_count = 0
        self.last_hh = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        sample_idx = self.np_random.integers(self.dataset.__len__())
        self.static, self.dynamic, _ = self.dataset[sample_idx]
        self.static = self.static.astype(np.float32)
        self.dynamic = self.dynamic.astype(np.float32)

        self.t = self.static[:4, :self.num_tasks].copy()
        self.mask = np.ones(self.num_nodes, dtype=np.float32)
        self.selected_nodes = []
        self.task_counts = np.zeros(self.num_tasks)
        self.episode_values = np.zeros(self.max_steps, dtype=np.float32)
        self.step_count = 0
        self.last_hh = None
        
        # 设置初始mask，排除距离过远的子系统
        coord1 = self.static[11:, :]
        coord2 = self.static[9:11, :]
        dist = np.sqrt(np.sum(np.square(coord1 - coord2), axis=0))
        self.mask = (dist <= self.distcon).astype(np.float32)

        observation = {
            "static": self.static,
            "dynamic": self.dynamic,
            "mask": self.mask,
            "t": self.t,
            "step_count": self.step_count,
        }
        info = {
            "task_counts": self.task_counts.copy(),
            "selected_nodes": self.selected_nodes.copy(),
            "mask_sum": float(self.mask.sum()),
            "t_sum": float(self.t.sum()),
            "step_count": self.step_count,
        }

        return observation, info
    
    def step(self, action):
        action = int(action)
        # print(f"Selected action: {action} as chosen_idx")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}, must be in [0, {self.num_nodes - 1}]")

        if self.mask[action] == 0:
            reward = 0.0
            observation = {
                "static": self.static,
                "dynamic": self.dynamic,
                "mask": self.mask,
                "t": self.t,
                "step_count": self.step_count,
            }
            return observation, reward, True, False, {"invalid": True}

        task_id = action % self.num_tasks
        self.task_counts[task_id] += 1

        decay = self.const1 ** self.task_counts[task_id] # 衰减因子
        self.t[:, task_id] -= self.static[4:8, action] * decay # 更新任务需求
        self.t[:, task_id] = np.maximum(self.t[:, task_id], 0)

        if self.step_count < self.max_steps:
            self.episode_values[self.step_count] = self.static[8, action]
        self.selected_nodes.append(action)
        self.step_count += 1
        # 完成的任务对应的所有节点设为不可选
        task_completed = self.t[:, task_id].sum() == 0
        if task_completed:
            self.mask[task_id::self.num_tasks] = 0

        system_id = action // self.num_tasks
        system_start = system_id * self.num_tasks
        system_end = system_start + self.num_tasks
        self.mask[system_start:system_end] = 0

        all_completed = self.t.sum() == 0
        no_valid_actions = self.mask.sum() == 0
        # 如果所有任务完成或没有可选动作，则结束
        terminated = all_completed or no_valid_actions
        # 如果达到最大步数，则截断
        truncated = self.step_count >= self.max_steps

        reward = 0.0
        if terminated or truncated:
            pair_rewards = np.sum(self.task_counts * (self.task_counts - 1) / 2)
            value_sum = self.episode_values[:self.step_count].sum()
            reward = value_sum / 20.0 + pair_rewards / 200.0

        observation = {
            "static": self.static,
            "dynamic": self.dynamic,
            "mask": self.mask,
            "t": self.t,
            "step_count": self.step_count,
        }
        info = {
            "task_counts": self.task_counts.copy(),
            "selected_nodes": self.selected_nodes.copy(),
            "mask_sum": float(self.mask.sum()),
            "t_sum": float(self.t.sum()),
            "step_count": self.step_count,
        }

        return observation, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        if mode != 'human':
            return None

        if self.static is None or len(self.selected_nodes) == 0:
            print("No data to render.")
            return None

        plt.close('all')

        coords = self.static[9:11, :]
        tour_indices = np.array(self.selected_nodes, dtype=np.int64)
        tour_coords = coords[:, tour_indices]
        # 绘制
        plt.figure(figsize=(6, 6))
        plt.scatter(coords[0], coords[1], s=6, c='gray', alpha=0.6, zorder=1)
        plt.plot(tour_coords[0], tour_coords[1], zorder=2)
        plt.scatter(tour_coords[0], tour_coords[1], s=10, c='r', zorder=3)
        plt.scatter(tour_coords[0, 0], tour_coords[1, 0], s=30, c='k', marker='*', zorder=4)

        plt.tight_layout()
        plt.savefig('sos_env_render.png', bbox_inches='tight', dpi=200)
        plt.close()

        return 'sos_env_render.png'

    def close(self):
        plt.close('all')