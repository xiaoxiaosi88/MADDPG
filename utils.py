import os
import numpy as np
import torch
import random
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories(config):
    """创建必要的目录"""
    dirs = [
        config.LOG_DIR,
        f"{config.LOG_DIR}/models",
        f"{config.LOG_DIR}/plots",
        f"{config.LOG_DIR}/videos"
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def save_config(config, filepath):
    """保存配置文件"""
    config_dict = {}
    for attr in dir(config):
        if not attr.startswith('__'):
            value = getattr(config, attr)
            if isinstance(value, (int, float, str, bool, list, dict)):
                config_dict[attr] = value
            elif isinstance(value, np.ndarray):
                config_dict[attr] = value.tolist()

    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=4)


class MovingAverage:
    """移动平均计算器"""

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.values = []

    def update(self, value):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)

    def get_average(self):
        if len(self.values) == 0:
            return 0
        return sum(self.values) / len(self.values)


def plot_training_curves(metrics, save_path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 奖励曲线
    axes[0, 0].plot(metrics['episode_rewards'])
    axes[0, 0].plot(metrics['episode_rewards_smooth'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend(['Raw', 'Smoothed'])
    axes[0, 0].grid(True)

    # 定位误差
    axes[0, 1].plot(metrics['localization_errors'])
    axes[0, 1].plot(metrics['localization_errors_smooth'])
    axes[0, 1].set_title('Localization Error')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Mean Error')
    axes[0, 1].legend(['Raw', 'Smoothed'])
    axes[0, 1].grid(True)

    # Actor损失
    if 'actor_losses' in metrics:
        axes[1, 0].plot(metrics['actor_losses'])
        axes[1, 0].set_title('Actor Loss')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)

    # Critic损失
    if 'critic_losses' in metrics:
        axes[1, 1].plot(metrics['critic_losses'])
        axes[1, 1].set_title('Critic Loss')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_localization_results(true_positions, estimated_positions,
                              anchor_positions, save_path):
    """绘制定位结果"""
    plt.figure(figsize=(12, 10))

    # 绘制真实位置
    plt.scatter(true_positions[:, 0], true_positions[:, 1],
                c='blue', marker='o', s=100, label='True Positions', alpha=0.7)

    # 绘制估计位置
    plt.scatter(estimated_positions[:, 0], estimated_positions[:, 1],
                c='red', marker='x', s=100, label='Estimated Positions')

    # 绘制锚点
    plt.scatter(anchor_positions[:, 0], anchor_positions[:, 1],
                c='green', marker='s', s=150, label='Anchors')

    # 绘制误差线
    for i in range(len(true_positions)):
        plt.plot([true_positions[i, 0], estimated_positions[i, 0]],
                 [true_positions[i, 1], estimated_positions[i, 1]],
                 'orange', alpha=0.5, linewidth=2)

    # 添加传感器编号
    for i in range(len(true_positions)):
        plt.annotate(f'S{i}', true_positions[i], xytext=(5, 5),
                     textcoords='offset points', fontsize=10)

    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Sensor Network Localization Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def calculate_localization_metrics(true_positions, estimated_positions):
    """计算定位性能指标"""
    errors = np.linalg.norm(estimated_positions - true_positions, axis=1)

    metrics = {
        'mean_error': np.mean(errors),
        'max_error': np.max(errors),
        'min_error': np.min(errors),
        'std_error': np.std(errors),
        'rmse': np.sqrt(np.mean(errors ** 2)),
        'errors': errors
    }

    return metrics


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=20, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, score, model=None):
        if self.best_score is None:
            self.best_score = score
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if model is not None and self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()

        return False


def get_device():
    """获取可用设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device