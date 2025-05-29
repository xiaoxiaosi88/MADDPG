import os
import json
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import logging


class TensorBoardLogger:
    """TensorBoard日志记录器"""

    def __init__(self, log_dir, experiment_name):
        self.log_dir = log_dir
        self.experiment_name = experiment_name

        # 创建带时间戳的日志目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.full_log_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")

        self.writer = SummaryWriter(self.full_log_dir)

        # 设置标量计数器
        self.episode_count = 0
        self.step_count = 0

    def log_scalar(self, tag, value, step=None):
        """记录标量值"""
        if step is None:
            step = self.step_count
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag, value_dict, step=None):
        """记录多个标量值"""
        if step is None:
            step = self.step_count
        self.writer.add_scalars(tag, value_dict, step)

    def log_histogram(self, tag, values, step=None):
        """记录直方图"""
        if step is None:
            step = self.step_count
        self.writer.add_histogram(tag, values, step)

    def log_episode_metrics(self, metrics):
        """记录episode级别的指标"""
        for key, value in metrics.items():
            self.log_scalar(f"Episode/{key}", value, self.episode_count)
        self.episode_count += 1

    def log_training_metrics(self, metrics):
        """记录训练级别的指标"""
        for key, value in metrics.items():
            self.log_scalar(f"Training/{key}", value, self.step_count)
        self.step_count += 1

    def log_agent_metrics(self, agent_id, metrics):
        """记录单个智能体的指标"""
        for key, value in metrics.items():
            self.log_scalar(f"Agent_{agent_id}/{key}", value, self.step_count)

    def log_network_weights(self, model, model_name):
        """记录网络权重分布"""
        for name, param in model.named_parameters():
            self.log_histogram(f"{model_name}/{name}", param.data, self.step_count)

    def close(self):
        """关闭writer"""
        self.writer.close()


class ExperimentLogger:
    """实验日志记录器"""

    def __init__(self, log_dir, experiment_name):
        self.log_dir = log_dir
        self.experiment_name = experiment_name

        # 创建日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")

        # 设置logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)

        # 指标存储
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'localization_errors': [],
            'actor_losses': [],
            'critic_losses': [],
            'noise_scales': []
        }

    def log_info(self, message):
        """记录信息"""
        self.logger.info(message)

    def log_episode(self, episode, episode_reward, episode_length,
                    localization_error, additional_metrics=None):
        """记录episode信息"""
        self.metrics['episode_rewards'].append(episode_reward)
        self.metrics['episode_lengths'].append(episode_length)
        self.metrics['localization_errors'].append(localization_error)

        # 计算移动平均
        recent_rewards = self.metrics['episode_rewards'][-100:]
        avg_reward = np.mean(recent_rewards)

        recent_errors = self.metrics['localization_errors'][-100:]
        avg_error = np.mean(recent_errors)

        message = (f"Episode {episode}: "
                   f"Reward={episode_reward:.2f}, "
                   f"Length={episode_length}, "
                   f"Error={localization_error:.4f}, "
                   f"Avg_Reward={avg_reward:.2f}, "
                   f"Avg_Error={avg_error:.4f}")

        if additional_metrics:
            for key, value in additional_metrics.items():
                message += f", {key}={value:.4f}"

        self.log_info(message)

    def log_training_step(self, actor_loss, critic_loss, noise_scale):
        """记录训练步骤信息"""
        self.metrics['actor_losses'].append(actor_loss)
        self.metrics['critic_losses'].append(critic_loss)
        self.metrics['noise_scales'].append(noise_scale)

    def save_metrics(self):
        """保存指标到文件"""
        metrics_file = os.path.join(self.log_dir, f"{self.experiment_name}_metrics.json")

        # 转换numpy数组为列表
        serializable_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            else:
                serializable_metrics[key] = value

        with open(metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)

    def get_metrics(self):
        """获取当前指标"""
        return self.metrics.copy()


class ProgressTracker:
    """训练进度跟踪器"""

    def __init__(self, total_episodes, log_interval=100):
        self.total_episodes = total_episodes
        self.log_interval = log_interval
        self.start_time = datetime.now()
        self.best_reward = float('-inf')
        self.best_error = float('inf')

    def update(self, episode, reward, error):
        """更新进度"""
        if reward > self.best_reward:
            self.best_reward = reward

        if error < self.best_error:
            self.best_error = error

        if episode % self.log_interval == 0:
            elapsed_time = datetime.now() - self.start_time
            progress = episode / self.total_episodes * 100

            print(f"\n{'=' * 60}")
            print(f"Progress: {progress:.1f}% ({episode}/{self.total_episodes})")
            print(f"Elapsed Time: {elapsed_time}")
            print(f"Best Reward: {self.best_reward:.2f}")
            print(f"Best Error: {self.best_error:.4f}")
            print(f"{'=' * 60}\n")