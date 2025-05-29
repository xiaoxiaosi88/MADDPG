import torch
import numpy as np
import os
import sys
from datetime import datetime

# 导入自定义模块
from MAL_env import SensorNetworkLocalizationEnv
from agent import MADDPGAgent
from memory import ReplayBuffer
from logger import TensorBoardLogger, ExperimentLogger, ProgressTracker
from utils import set_seed, create_directories, save_config, MovingAverage, \
    plot_training_curves, plot_localization_results, \
    calculate_localization_metrics, get_device
import config


class MADDPGTrainer:
    """MADDPG训练器"""

    def __init__(self, config):
        self.config = config
        self.device = get_device()

        # 设置随机种子
        set_seed(config.SEED)

        # 创建目录
        create_directories(config)

        # 创建环境
        self.env = SensorNetworkLocalizationEnv(
            anchors_pos=config.ANCHORS_POS,
            sensor_pos=config.SENSORS_POS,
            estimated_positions=config.ESTIMATED_POSITIONS,
            communication_range=config.COMMUNICATION_RANGE,
            noise_std=config.NOISE_STD,
            max_episode_steps=config.MAX_EPISODE_STEPS,
            initial_pos_bounds=config.INITIAL_POS_BOUNDS_IVA,
            render_mode=config.RENDER_MODE,
            dimension=config.DIMENSION
        )

        # 获取环境信息
        self.n_agents = len(self.env.agents)
        self.agent_names = self.env.agents

        # 获取状态和动作维度 - 修复错误1: 正确获取维度
        sample_obs = self.env.reset()
        self.state_dim = sample_obs[self.agent_names[0]].shape[0]
        # 修复错误2: 正确访问action_space
        self.action_dim = self.env.action_space[self.agent_names[0]].shape[0]

        # 计算全局状态和动作维度
        self.global_state_dim = self.env.get_global_state_dim()
        self.global_action_dim = self.env.get_joint_action_space_size()

        # 创建智能体
        self.agents = {}
        for i, agent_name in enumerate(self.agent_names):
            self.agents[agent_name] = MADDPGAgent(
                agent_id=i,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                global_state_dim=self.global_state_dim,
                global_action_dim=self.global_action_dim,
                config=config,
                device=self.device
            )

        # 创建经验回放缓冲区
        self.replay_buffer = ReplayBuffer(config.MEMORY_SIZE, self.n_agents)

        # 创建日志记录器
        self.tb_logger = TensorBoardLogger(config.LOG_DIR, config.EXPERIMENT_NAME)
        self.exp_logger = ExperimentLogger(config.LOG_DIR, config.EXPERIMENT_NAME)
        self.progress_tracker = ProgressTracker(config.TOTAL_TIMESTEPS // config.MAX_EPISODE_STEPS)

        # 移动平均计算器
        self.reward_avg = MovingAverage(100)
        self.error_avg = MovingAverage(100)

        # 训练统计
        self.total_steps = 0
        self.episode_count = 0

        # 保存配置
        save_config(config, os.path.join(config.LOG_DIR, "config.json"))

        self.exp_logger.log_info("MADDPG Trainer initialized successfully")
        self.exp_logger.log_info(f"Environment: {self.n_agents} agents, "
                                 f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        self.exp_logger.log_info(f"Global state dim: {self.global_state_dim}, "
                                 f"Global action dim: {self.global_action_dim}")

    def get_global_state(self, observations):
        """构建全局状态"""
        # 使用环境的内置方法
        return self.env.get_global_state()

    def get_global_action(self, actions):
        """构建全局动作"""
        # 使用环境的内置方法
        return self.env.get_global_action_from_dict(actions)

    def collect_experience(self):
        """收集一个episode的经验"""
        observations = self.env.reset()  # 修复错误3: 环境返回单个值
        episode_reward = 0
        episode_length = 0
        info = {}  # 修复错误4: 初始化info变量

        # 重置智能体噪声
        for agent in self.agents.values():
            agent.reset_noise()

        for step in range(self.config.MAX_EPISODE_STEPS):
            # 获取所有智能体的动作
            actions = {}
            for agent_name, agent in self.agents.items():
                action = agent.act(observations[agent_name], add_noise=True)
                actions[agent_name] = action

            # 构建全局状态和动作
            global_state = self.get_global_state(observations)
            global_action = self.get_global_action(actions)

            # 环境步进 - 修复错误5: 正确解包返回值
            step_result = self.env.step(actions)
            if len(step_result) == 4:
                next_observations, rewards, done, info = step_result
            else:
                # 处理可能的不同返回格式
                next_observations = step_result[0]
                rewards = step_result[1]
                done = step_result[2] if len(step_result) > 2 else {'__all__': False}
                info = step_result[3] if len(step_result) > 3 else {}

            # 构建下一个全局状态
            next_global_state = self.env.get_global_state()

            # 检查终止条件
            is_done = done.get('__all__', False) if isinstance(done, dict) else done

            # 存储经验
            agent_states = [observations[name] for name in self.agent_names]
            agent_actions = [actions[name] for name in self.agent_names]
            agent_rewards = [rewards[name] for name in self.agent_names]
            agent_next_states = [next_observations[name] for name in self.agent_names]
            agent_dones = [is_done for _ in self.agent_names]

            self.replay_buffer.push(
                agent_states, agent_actions, agent_rewards,
                agent_next_states, agent_dones,
                global_state, global_action, next_global_state
            )

            # 更新状态
            observations = next_observations
            episode_reward += sum(rewards.values())
            episode_length += 1
            self.total_steps += 1

            # 训练
            if (len(self.replay_buffer) > self.config.LEARNING_STARTS and
                    self.total_steps % self.config.POLICY_FREQUENCY == 0):
                self.train_agents()

            if is_done:
                break

        # 获取定位误差
        localization_error = info.get('mean_localization_error', 0.0)

        return episode_reward, episode_length, localization_error, info

    def train_agents(self):
        """训练所有智能体"""
        if len(self.replay_buffer) < self.config.BATCH_SIZE:
            return

        # 采样批次数据
        batch = self.replay_buffer.sample(self.config.BATCH_SIZE, self.device)

        # 计算下一步的全局动作 (使用目标网络)
        next_global_actions = []
        for i, agent in enumerate(self.agents.values()):
            next_action = agent.actor_target(batch['agent_next_states'][:, i])
            next_global_actions.append(next_action)
        next_global_actions = torch.cat(next_global_actions, dim=1)

        # 训练每个智能体
        total_actor_loss = 0
        total_critic_loss = 0

        for agent in self.agents.values():
            # 更新Critic
            critic_loss = agent.update_critic(batch, next_global_actions)
            total_critic_loss += critic_loss

            # 更新Actor
            actor_loss = agent.update_actor(batch, list(self.agents.values()))
            total_actor_loss += actor_loss

            # 更新目标网络
            if self.total_steps % self.config.TARGET_NETWORK_FREQUENCY == 0:
                agent.update_target_networks()

        # 记录训练指标
        avg_actor_loss = total_actor_loss / self.n_agents
        avg_critic_loss = total_critic_loss / self.n_agents
        avg_noise_scale = np.mean([agent.noise_scale for agent in self.agents.values()])

        self.exp_logger.log_training_step(avg_actor_loss, avg_critic_loss, avg_noise_scale)

        # TensorBoard记录
        self.tb_logger.log_training_metrics({
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'noise_scale': avg_noise_scale,
            'buffer_size': len(self.replay_buffer)
        })

    def train(self):
        """主训练循环"""
        self.exp_logger.log_info("Starting MADDPG training...")

        while self.total_steps < self.config.TOTAL_TIMESTEPS:
            # 收集经验并训练
            episode_reward, episode_length, localization_error, info = self.collect_experience()

            self.episode_count += 1

            # 更新移动平均
            self.reward_avg.update(episode_reward)
            self.error_avg.update(localization_error)

            # 衰减噪声
            for agent in self.agents.values():
                agent.decay_noise()

            # 记录episode指标
            episode_metrics = {
                'reward': episode_reward,
                'length': episode_length,
                'localization_error': localization_error,
                'reward_avg': self.reward_avg.get_average(),
                'error_avg': self.error_avg.get_average(),
                'total_steps': self.total_steps
            }

            self.tb_logger.log_episode_metrics(episode_metrics)
            self.exp_logger.log_episode(
                self.episode_count, episode_reward, episode_length,
                localization_error, {
                    'avg_reward': self.reward_avg.get_average(),
                    'avg_error': self.error_avg.get_average()
                }
            )

            # 更新进度
            self.progress_tracker.update(self.episode_count, episode_reward, localization_error)

            # 定期保存模型和绘图
            if self.episode_count % 100 == 0:
                self.save_models()
                self.plot_results()

            # 定期渲染环境
            if self.episode_count % 50 == 0:
                self.render_episode()

        self.exp_logger.log_info("Training completed!")
        self.finalize_training()

    def save_models(self):
        """保存所有智能体的模型"""
        if not self.config.SAVE_MODEL:
            return

        models_dir = os.path.join(self.config.LOG_DIR, "models")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for agent_name, agent in self.agents.items():
            model_path = os.path.join(models_dir, f"{agent_name}_{timestamp}.pth")
            agent.save_models(model_path)

        self.exp_logger.log_info(f"Models saved at episode {self.episode_count}")

    def plot_results(self):
        """绘制训练结果"""
        metrics = self.exp_logger.get_metrics()

        # 计算平滑曲线
        if len(metrics['episode_rewards']) > 10:
            window = min(50, len(metrics['episode_rewards']) // 4)

            metrics['episode_rewards_smooth'] = []
            metrics['localization_errors_smooth'] = []

            for i in range(len(metrics['episode_rewards'])):
                start_idx = max(0, i - window)
                end_idx = i + 1

                avg_reward = np.mean(metrics['episode_rewards'][start_idx:end_idx])
                avg_error = np.mean(metrics['localization_errors'][start_idx:end_idx])

                metrics['episode_rewards_smooth'].append(avg_reward)
                metrics['localization_errors_smooth'].append(avg_error)

        # 绘制训练曲线
        plot_path = os.path.join(self.config.LOG_DIR, "plots",
                                 f"training_curves_ep{self.episode_count}.png")
        plot_training_curves(metrics, plot_path)

    def render_episode(self):
        """渲染一个测试episode"""
        # 设置为评估模式
        for agent in self.agents.values():
            agent.set_training_mode(False)

        observations = self.env.reset()
        info = {}  # 初始化info

        for step in range(self.config.MAX_EPISODE_STEPS):
            actions = {}
            for agent_name, agent in self.agents.items():
                action = agent.act(observations[agent_name], add_noise=False)
                actions[agent_name] = action

            step_result = self.env.step(actions)
            if len(step_result) == 4:
                observations, rewards, done, info = step_result
            else:
                observations = step_result[0]
                rewards = step_result[1]
                done = step_result[2] if len(step_result) > 2 else {'__all__': False}
                info = step_result[3] if len(step_result) > 3 else {}

            is_done = done.get('__all__', False) if isinstance(done, dict) else done
            if is_done:
                break

        # 绘制定位结果
        true_pos = info.get('true_positions', self.env.true_positions)
        est_pos = info.get('estimated_positions', self.env.estimated_positions)
        anchor_pos = self.config.ANCHORS_POS

        plot_path = os.path.join(self.config.LOG_DIR, "plots",
                                 f"localization_ep{self.episode_count}.png")
        plot_localization_results(true_pos, est_pos, anchor_pos, plot_path)

        # 计算定位指标
        loc_metrics = calculate_localization_metrics(true_pos, est_pos)
        self.tb_logger.log_scalars("Localization", loc_metrics, self.episode_count)

        # 恢复训练模式
        for agent in self.agents.values():
            agent.set_training_mode(True)

    def finalize_training(self):
        """完成训练后的清理工作"""
        # 保存最终模型
        self.save_models()

        # 保存指标
        self.exp_logger.save_metrics()

        # 最终结果图
        self.plot_results()
        self.render_episode()

        # 关闭日志记录器
        self.tb_logger.close()

        self.exp_logger.log_info("All training artifacts saved successfully!")


def main():
    """主函数"""
    print("Starting MADDPG Training for Sensor Network Localization")
    print("=" * 60)

    # 创建训练器并开始训练
    trainer = MADDPGTrainer(config)
    trainer.train()

    print("Training completed successfully!")


if __name__ == "__main__":
    main()