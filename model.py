import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Actor(nn.Module):
    """MADDPG Actor网络 - 确定性策略"""

    def __init__(self, state_dim, action_dim, hidden_units=256, max_action=1.0):
        super(Actor, self).__init__()
        self.max_action = max_action

        self.fc1 = nn.Linear(state_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.fc4 = nn.Linear(hidden_units, action_dim)

        self.dropout = nn.Dropout(0.1)

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """初始化网络权重"""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

        # 最后一层使用较小的初始化
        nn.init.uniform_(self.fc4.weight, -3e-3, 3e-3)
        nn.init.constant_(self.fc4.bias, 0)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))

        return self.max_action * x


class Critic(nn.Module):
    """MADDPG Critic网络 - 中心化价值函数"""

    def __init__(self, global_state_dim, global_action_dim, hidden_units=256):
        super(Critic, self).__init__()

        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(global_state_dim, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_units, hidden_units // 2)
        )

        # 动作编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(global_action_dim, hidden_units // 2),
            nn.ReLU()
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.ReLU(),
            nn.Linear(hidden_units // 2, 1)
        )

        self.init_weights()

    def init_weights(self):
        """初始化网络权重"""
        for module in [self.state_encoder, self.action_encoder, self.fusion]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)

    def forward(self, global_state, global_action):
        state_features = self.state_encoder(global_state)
        action_features = self.action_encoder(global_action)

        # 特征融合
        x = torch.cat([state_features, action_features], dim=1)
        q_value = self.fusion(x)

        return q_value


class OUNoise:
    """Ornstein-Uhlenbeck噪声 - 用于连续动作空间的探索"""

    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        """重置噪声状态"""
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        """采样噪声"""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state = self.state + dx
        return self.state.copy()


def soft_update(target, source, tau):
    """软更新目标网络参数"""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    """硬更新目标网络参数"""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)