import gym
from gym import spaces
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class SensorNetworkLocalizationEnv(gym.Env):
    """
    多智能体传感器网络定位环境
    基于连续动作空间和局部奖励的强化学习环境
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

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
        super(SensorNetworkLocalizationEnv, self).__init__()
        self.anchors_pos = np.array(anchors_pos, dtype=np.float32)
        self.sensor_pos = np.array(sensor_pos, dtype=np.float32)
        self.n_sensors = self.sensor_pos.shape[0]
        self.n_anchors = self.anchors_pos.shape[0]

        # 重要修改1: 移除area_size，使用initial_pos_bounds
        self.initial_pos_bounds = initial_pos_bounds
        self.communication_range = communication_range
        self.noise_std = noise_std
        self.max_episode_steps = max_episode_steps
        self.dimension = dimension

        # MADDPG需要的智能体列表
        self.agents = [f'sensor_{i}' for i in range(self.n_sensors)]

        # 动作空间：连续位移向量 Δx_i ∈ ℝ^D
        self.action_space = spaces.Dict({
            f'sensor_{i}': spaces.Box(
                low=-300.0, high=300.0,
                shape=(dimension,), dtype=np.float32
            ) for i in range(self.n_sensors)
        })

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

        self.observation_space = spaces.Dict({
            f'sensor_{i}': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(state_dim_per_sensor,), dtype=np.float32
            ) for i in range(self.n_sensors)
        })

        # 环境状态
        self.render_mode = render_mode
        self.true_positions = None  # 真实位置
        self.estimated_positions = estimated_positions  # 估计位置
        self.anchor_positions = None  # 锚点位置
        self.communication_graph = None  # 通信图
        self.distance_measurements = None  # 距离测量
        self.anchor_measurements = None  # 锚点距离测量

        self.current_step = 0
        self.max_neighbors = max_neighbors
        self.max_anchors_per_sensor = max_anchors_per_sensor

    def reset(self, seed=None, options=None) -> Dict[str, np.ndarray]:
        """重置环境"""
        super().reset(seed=seed)
        self.current_step = 0

        # 生成真实位置
        self.true_positions = self.sensor_pos

        # 生成锚点位置
        self.anchor_positions = self.anchors_pos

        # 重要修改2: 初始化估计位置时考虑边界约束
        # 原来的代码可能导致估计位置超出合理范围
        initial_noise = np.random.normal(0, self.noise_std * 2, size=self.true_positions.shape)
        self.estimated_positions = self.true_positions + initial_noise

        # 确保初始估计位置在合理范围内
        # self.estimated_positions = self._clip_positions_to_bounds(self.estimated_positions)
        self.estimated_positions = self.estimated_positions

        # 构建通信图
        self.communication_graph = self._build_communication_graph()

        # 生成距离测量
        self._generate_measurements()

        return self._get_observations()

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict]:
        """执行一步动作"""
        self.current_step += 1

        # 更新位置估计：x̂_i^{t+1} = x̂_i^t + Δx_i
        for i in range(self.n_sensors):
            action = actions[f'sensor_{i}']
            self.estimated_positions[i] += action

        # 重要修改3: 使用边界约束而不是固定区域大小
        # 可选：如果需要约束位置在边界内，取消下面的注释
        # self.estimated_positions = self._clip_positions_to_bounds(self.estimated_positions)

        # 重新生成距离测量
        self._generate_measurements()

        # 计算奖励
        rewards = self._compute_local_rewards()

        # 检查终止条件
        done = self._check_done()

        # 获取新观测
        observations = self._get_observations()

        # 计算信息
        info = self._get_info()

        return observations, rewards, done, info

    # def _clip_positions_to_bounds(self, positions: np.ndarray) -> np.ndarray:
    #     """
    #     重要修改4: 新增方法 - 将位置约束在边界范围内
    #     """
    #     clipped_positions = positions.copy()
    #
    #     for dim in range(self.dimension):
    #         if dim < len(self.initial_pos_bounds):
    #             min_bound, max_bound = self.initial_pos_bounds[dim]
    #             clipped_positions[:, dim] = np.clip(
    #                 clipped_positions[:, dim], min_bound, max_bound
    #             )
    #
    #     return clipped_positions

    def _build_communication_graph(self) -> nx.Graph:
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

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """获取观测"""
        observations = {}

        for i in range(self.n_sensors):
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

            observations[f'sensor_{i}'] = state.astype(np.float32)

        return observations

    def get_global_state(self):
        """
        获取全局状态 - MADDPG的Critic需要
        返回所有智能体的观测和动作的组合
        """
        global_obs = []
        for agent in self.agents:
            agent_idx = int(agent.split('_')[1])
            global_obs.append(self.estimated_positions[agent_idx])

        # 构建全局状态，包含：
        # 1. 所有智能体的当前估计位置
        # 2. 所有锚点位置
        # 3. 所有智能体的真实位置（用于中心化训练）
        global_state = np.concatenate([
            np.array(global_obs).flatten(),  # 所有智能体估计位置
            self.anchor_positions.flatten(),  # 锚点位置
            self.true_positions.flatten(),  # 真实位置（训练时可用）
        ])

        return global_state

    def get_joint_action_space_size(self):
        """返回联合动作空间大小"""
        single_action_size = self.action_space[self.agents[0]].shape[0]
        return single_action_size * len(self.agents)

    def get_global_action_from_dict(self, actions_dict):
        """
        将字典形式的动作转换为全局动作向量

        Args:
            actions_dict: {agent_name: action_vector} 形式的动作字典

        Returns:
            np.ndarray: 全局动作向量
        """
        global_action = []
        for agent in self.agents:
            global_action.append(actions_dict[agent])
        return np.concatenate(global_action)

    def get_global_state_dim(self):
        """返回全局状态维度"""
        # 所有智能体估计位置 + 锚点位置 + 真实位置
        return (self.n_sensors * self.dimension +  # 估计位置
                self.n_anchors * self.dimension +  # 锚点位置
                self.n_sensors * self.dimension)  # 真实位置

    def get_local_observation_dim(self):
        """返回单个智能体的观测维度"""
        agent_name = self.agents[0]
        return self.observation_space[agent_name].shape[0]

    def _compute_local_rewards(self) -> Dict[str, float]:
        """计算局部奖励"""
        rewards = {}

        for i in range(self.n_sensors):
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
            rewards[f'sensor_{i}'] = reward

        return rewards

    def _check_done(self) -> Dict[str, bool]:
        """检查终止条件"""
        done = {'__all__': self.current_step >= self.max_episode_steps}

        # 可以添加其他终止条件，如收敛检查
        localization_error = np.mean(np.linalg.norm(
            self.estimated_positions - self.true_positions, axis=1
        ))

        if localization_error < 0.1:  # 收敛阈值
            done['__all__'] = True

        return done

    def _get_info(self) -> Dict:
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

    def render(self, mode='human'):
        """
        重要修改5: 更新渲染函数以使用边界而不是固定区域大小
        """
        if mode == 'human':
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

            # 重要修改6: 使用边界设置坐标轴范围
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
            plt.show()

    def close(self):
        """关闭环境"""
        pass


# 多智能体包装器 - 无需修改
class MultiAgentWrapper:
    """
    多智能体环境包装器，兼容主流MARL框架
    """

    def __init__(self, env):
        self.env = env
        self.agents = [f'sensor_{i}' for i in range(env.n_sensors)]

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        return self.env.step(actions)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space


# 重要修改7: 更新使用示例
if __name__ == "__main__":
    # 示例：定义锚点和传感器位置
    anchors_pos = np.array([[0.0, 0.0], [250.0, 0.0], [0.0, 200.0]])
    sensor_pos = np.random.uniform([50, 50], [200, 150], size=(8, 2))
    estimated_positions = sensor_pos + np.random.normal(0, 5, sensor_pos.shape)

    # 创建环境
    env = SensorNetworkLocalizationEnv(
        anchors_pos=anchors_pos,
        sensor_pos=sensor_pos,
        estimated_positions=estimated_positions,
        communication_range=60.0,
        noise_std=1.0,
        initial_pos_bounds=np.array([[-50.0, 300.0], [-50.0, 250.0]])
    )

    # 包装为多智能体环境
    multi_env = MultiAgentWrapper(env)

    # 测试环境
    obs = multi_env.reset()
    print("观测空间维度:", {k: v.shape for k, v in list(obs.items())[:3]})  # 只显示前3个
    print("动作空间:", {k: v for k, v in list(multi_env.action_space.spaces.items())[:3]})

    # 测试MADDPG需要的功能
    print(f"全局状态维度: {env.get_global_state_dim()}")
    print(f"联合动作空间大小: {env.get_joint_action_space_size()}")
    print(f"单智能体观测维度: {env.get_local_observation_dim()}")

    global_state = env.get_global_state()
    print(f"当前全局状态形状: {global_state.shape}")

    # 随机动作测试
    for step in range(3):
        actions = {}
        for agent in multi_env.agents:
            actions[agent] = multi_env.action_space[agent].sample() * 0.1  # 减小随机动作幅度

        obs, rewards, done, info = multi_env.step(actions)

        # 测试全局动作转换
        global_action = env.get_global_action_from_dict(actions)
        print(f"Step {step + 1}:")
        print(f"  平均定位误差: {info['mean_localization_error']:.4f}")
        print(f"  前3个智能体奖励: {[(k, f'{v:.2f}') for k, v in list(rewards.items())[:3]]}")
        print(f"  全局动作形状: {global_action.shape}")

        if done['__all__']:
            break

    # 渲染环境
    multi_env.render()
