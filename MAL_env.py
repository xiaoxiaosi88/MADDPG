import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces


class SensorNetworkLocalizationEnv(ParallelEnv):
    """
    多智能体传感器网络定位环境
    基于连续动作空间和局部奖励的强化学习环境
    直接实现PettingZoo的ParallelEnv接口
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self,
                 anchors_pos: np.ndarray,
                 sensor_pos: np.ndarray,
                 estimated_positions: np.ndarray,
                 communication_range: float = 3.0,
                 noise_std: float = 0.1,
                 max_episode_steps: int = 100,
                 initial_pos_bounds: np.ndarray = np.array([[-50.0, 300.0], [-50.0, 250.0]]),
                 render_mode: str = None,
                 dimension: int = 2):
        """
        初始化传感器网络定位环境

        Args:
            anchors_pos: 锚点位置
            sensor_pos: 传感器真实位置
            estimated_positions: 初始估计位置
            communication_range: 通信范围
            noise_std: 噪声标准差
            max_episode_steps: 最大步数
            initial_pos_bounds: 位置边界范围 [[x_min, x_max], [y_min, y_max]]
            render_mode: 渲染模式
            dimension: 空间维度（2D或3D）
        """
        self.anchors_pos = np.array(anchors_pos, dtype=np.float32)
        self.sensor_pos = np.array(sensor_pos, dtype=np.float32)
        self.n_sensors = self.sensor_pos.shape[0]
        self.n_anchors = self.anchors_pos.shape[0]

        # PettingZoo必要的属性
        self.possible_agents = [f'sensor_{i}' for i in range(self.n_sensors)]
        self.agents = self.possible_agents.copy()

        self.initial_pos_bounds = initial_pos_bounds
        self.communication_range = communication_range
        self.noise_std = noise_std
        self.max_episode_steps = max_episode_steps
        self.dimension = dimension
        self.render_mode = render_mode

        # 动作空间和观察空间定义为字典
        self._action_spaces = {
            agent: spaces.Box(
                low=-300.0, high=300.0,
                shape=(dimension,), dtype=np.float32
            ) for agent in self.possible_agents
        }

        # 状态空间维度计算
        max_neighbors = self.n_sensors - 1  # 最大邻居数
        max_anchors_per_sensor = self.n_anchors  # 最大锚点数

        state_dim_per_sensor = (
                dimension +  # 当前位置估计
                max_neighbors * dimension +  # 邻居位置
                max_neighbors +  # 邻居距离
                max_anchors_per_sensor +  # 锚点距离
                max_anchors_per_sensor * dimension +  # 锚点位置
                max_neighbors +  # 邻居掩码
                max_anchors_per_sensor  # 锚点掩码
        )

        self._observation_spaces = {
            agent: spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(state_dim_per_sensor,), dtype=np.float32
            ) for agent in self.possible_agents
        }

        # 环境状态
        self.true_positions = None  # 真实位置
        self.estimated_positions = estimated_positions  # 估计位置
        self.anchor_positions = None  # 锚点位置
        self.communication_graph = None  # 通信图
        self.distance_measurements = None  # 距离测量
        self.anchor_measurements = None  # 锚点距离测量

        self.current_step = 0
        self.max_neighbors = max_neighbors
        self.max_anchors_per_sensor = max_anchors_per_sensor

    def observation_space(self, agent):
        """返回指定智能体的观察空间"""
        return self._observation_spaces[agent]

    def action_space(self, agent):
        """返回指定智能体的动作空间"""
        return self._action_spaces[agent]

    def state(self):
        """
        返回环境的全局状态
        用于中心化训练
        """
        return np.concatenate([
            self.true_positions.flatten(),
            self.estimated_positions.flatten(),
            self.anchor_positions.flatten()
        ])

    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0

        # 生成真实位置
        self.true_positions = self.sensor_pos

        # 生成锚点位置
        self.anchor_positions = self.anchors_pos

        # 初始化估计位置
        initial_noise = np.random.normal(0, self.noise_std * 2, size=self.true_positions.shape)
        self.estimated_positions = self.true_positions + initial_noise

        # 构建通信图
        self.communication_graph = self._build_communication_graph()

        # 生成距离测量
        self._generate_measurements()

        observations = self._get_observations()
        info = {}
        return observations, info

    def step(self, actions):
        """执行一步动作"""
        self.current_step += 1

        # 更新位置估计：x̂_i^{t+1} = x̂_i^t + Δx_i
        for i, agent in enumerate(self.agents):
            action = actions[agent]
            self.estimated_positions[i] += action

        # 重新生成距离测量
        self._generate_measurements()

        # 计算奖励
        rewards = self._compute_local_rewards()

        # 获取新观测
        observations = self._get_observations()

        # 检查终止条件
        terminated = self._check_terminated()
        truncated = self._check_truncated()

        # 计算信息
        info = self._get_info()

        return observations, rewards, terminated, truncated, info

    def _build_communication_graph(self):
        """构建通信图"""
        G = nx.Graph()
        G.add_nodes_from(range(self.n_sensors))

        for i in range(self.n_sensors):
            for j in range(i + 1, self.n_sensors):
                distance = np.linalg.norm(
                    self.true_positions[i] - self.true_positions[j]
                )
                if distance <= self.communication_range:
                    G.add_edge(i, j)

        return G

    def _generate_measurements(self):
        """生成距离测量"""
        # 传感器间距离测量
        self.distance_measurements = {}
        for edge in self.communication_graph.edges():
            i, j = edge
            true_distance = np.linalg.norm(
                self.true_positions[i] - self.true_positions[j]
            )
            noise = np.random.normal(0, self.noise_std)
            self.distance_measurements[(i, j)] = true_distance + noise

        # 锚点距离测量
        self.anchor_measurements = {}
        for i in range(self.n_sensors):
            for k in range(self.n_anchors):
                distance = np.linalg.norm(
                    self.true_positions[i] - self.anchor_positions[k]
                )
                if distance <= self.communication_range * 1.5:  # 锚点通信范围更大
                    noise = np.random.normal(0, self.noise_std)
                    self.anchor_measurements[(i, k)] = distance + noise

    def _get_observations(self):
        """获取观测"""
        observations = {}

        for i, agent in enumerate(self.agents):
            # 当前位置估计
            current_pos = self.estimated_positions[i]

            # 获取邻居信息
            neighbors = list(self.communication_graph.neighbors(i))
            neighbor_positions = np.zeros((self.max_neighbors, self.dimension))
            neighbor_distances = np.zeros(self.max_neighbors)
            neighbor_mask = np.zeros(self.max_neighbors)

            for idx, neighbor in enumerate(neighbors[:self.max_neighbors]):
                neighbor_positions[idx] = self.estimated_positions[neighbor]
                # 使用测量距离而不是估计距离
                if (i, neighbor) in self.distance_measurements:
                    neighbor_distances[idx] = self.distance_measurements[(i, neighbor)]
                elif (neighbor, i) in self.distance_measurements:
                    neighbor_distances[idx] = self.distance_measurements[(neighbor, i)]
                neighbor_mask[idx] = 1.0

            # 获取锚点信息
            anchor_distances = np.zeros(self.max_anchors_per_sensor)
            anchor_positions = np.zeros((self.max_anchors_per_sensor, self.dimension))
            anchor_mask = np.zeros(self.max_anchors_per_sensor)

            anchor_idx = 0
            for k in range(self.n_anchors):
                if (i, k) in self.anchor_measurements and anchor_idx < self.max_anchors_per_sensor:
                    anchor_distances[anchor_idx] = self.anchor_measurements[(i, k)]
                    anchor_positions[anchor_idx] = self.anchor_positions[k]
                    anchor_mask[anchor_idx] = 1.0
                    anchor_idx += 1

            # 组合状态
            state = np.concatenate([
                current_pos,
                neighbor_positions.flatten(),
                neighbor_distances,
                anchor_distances,
                anchor_positions.flatten(),
                neighbor_mask,
                anchor_mask
            ])

            observations[agent] = state.astype(np.float32)

        return observations

    def _compute_local_rewards(self):
        """计算局部奖励"""
        rewards = {}

        for i, agent in enumerate(self.agents):
            reward1 = 0.0
            reward2 = 0.0

            # 与邻居的测量误差
            neighbors = list(self.communication_graph.neighbors(i))
            for neighbor in neighbors:
                estimated_distance = np.linalg.norm(
                    self.estimated_positions[i] - self.estimated_positions[neighbor]
                )

                if (i, neighbor) in self.distance_measurements:
                    measured_distance = self.distance_measurements[(i, neighbor)]
                elif (neighbor, i) in self.distance_measurements:
                    measured_distance = self.distance_measurements[(neighbor, i)]
                else:
                    continue

                # 局部奖励：负测量误差平方
                error = (estimated_distance - measured_distance) ** 2
                reward1 -= error / (self.noise_std ** 2)

            # 与锚点的测量误差
            for k in range(self.n_anchors):
                if (i, k) in self.anchor_measurements:
                    estimated_distance = np.linalg.norm(
                        self.estimated_positions[i] - self.anchor_positions[k]
                    )
                    measured_distance = self.anchor_measurements[(i, k)]

                    error = (estimated_distance - measured_distance) ** 2
                    reward2 -= error / (self.noise_std ** 2)

            reward = reward1 + reward2
            rewards[agent] = reward

        return rewards

    def _check_terminated(self):
        """检查是否终止"""
        # 收敛检查
        terminated = {agent: False for agent in self.agents}

        localization_error = np.mean(np.linalg.norm(
            self.estimated_positions - self.true_positions, axis=1
        ))

        if localization_error < 0.1:  # 收敛阈值
            terminated = {agent: True for agent in self.agents}
            terminated["__all__"] = True
        else:
            terminated["__all__"] = False

        return terminated

    def _check_truncated(self):
        """检查是否因为步数限制而被截断"""
        truncated = {agent: False for agent in self.agents}

        if self.current_step >= self.max_episode_steps:
            truncated = {agent: True for agent in self.agents}
            truncated["__all__"] = True
        else:
            truncated["__all__"] = False

        return truncated

    def _get_info(self):
        """获取额外信息"""
        localization_errors = np.linalg.norm(
            self.estimated_positions - self.true_positions, axis=1
        )

        return {
            'mean_localization_error': np.mean(localization_errors),
            'max_localization_error': np.max(localization_errors),
            'step': self.current_step,
            'true_positions': self.true_positions.copy(),
            'estimated_positions': self.estimated_positions.copy()
        }

    def render(self):
        """渲染环境"""
        plt.figure(figsize=(12, 10))

        # 绘制真实位置
        plt.scatter(self.true_positions[:, 0], self.true_positions[:, 1],
                    c='blue', marker='o', s=100, label='True Positions', alpha=0.7)

        # 绘制估计位置
        plt.scatter(self.estimated_positions[:, 0], self.estimated_positions[:, 1],
                    c='red', marker='x', s=100, label='Estimated Positions')

        # 绘制锚点
        plt.scatter(self.anchor_positions[:, 0], self.anchor_positions[:, 1],
                    c='green', marker='s', s=150, label='Anchors')

        # 绘制通信链接
        for edge in self.communication_graph.edges():
            i, j = edge
            plt.plot([self.true_positions[i, 0], self.true_positions[j, 0]],
                     [self.true_positions[i, 1], self.true_positions[j, 1]],
                     'gray', alpha=0.3, linewidth=1)

        # 绘制误差线
        for i in range(self.n_sensors):
            plt.plot([self.true_positions[i, 0], self.estimated_positions[i, 0]],
                     [self.true_positions[i, 1], self.estimated_positions[i, 1]],
                     'orange', alpha=0.5, linewidth=2)

        # 使用边界设置坐标轴范围
        x_min, x_max = self.initial_pos_bounds[0]
        y_min, y_max = self.initial_pos_bounds[1]

        # 添加一些边距以便更好地观察
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1

        plt.xlim(x_min - x_margin, x_max + x_margin)
        plt.ylim(y_min - y_margin, y_max + y_margin)

        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title(f'Sensor Network Localization - Step {self.current_step}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'sensor_network_step_{self.current_step}.png')  # 保存图片
        plt.close()

    def close(self):
        """关闭环境"""
        pass