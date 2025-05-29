import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import Actor, Critic, OUNoise, soft_update, hard_update


class MADDPGAgent:
    """MADDPG智能体"""

    def __init__(self, agent_id, state_dim, action_dim, global_state_dim,
                 global_action_dim, config, device='cpu'):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim
        self.global_action_dim = global_action_dim
        self.device = device
        self.config = config

        # 动作范围 (根据您的环境配置)
        self.max_action = 300.0  # 对应您环境中的动作范围

        # Actor网络
        self.actor = Actor(state_dim, action_dim, config.HIDDEN_UNITS, self.max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, config.HIDDEN_UNITS, self.max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.ACTOR_LEARNING_RATE)

        # Critic网络
        self.critic = Critic(global_state_dim, global_action_dim, config.HIDDEN_UNITS).to(device)
        self.critic_target = Critic(global_state_dim, global_action_dim, config.HIDDEN_UNITS).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.CRITIC_LEARNING_RATE)

        # 初始化目标网络
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # 探索噪声
        self.noise = OUNoise(action_dim, sigma=config.NOISE_SCALE)
        self.noise_scale = config.NOISE_SCALE
        self.noise_decay = config.NOISE_DECAY
        self.min_noise = config.MIN_NOISE

        # 损失函数
        self.criterion = nn.MSELoss()

        # 训练模式标志
        self.training_mode = True

    def act(self, state, add_noise=True):
        """选择动作"""
        self.actor.eval()
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            elif len(state.shape) == 1:
                state = state.unsqueeze(0)

            action = self.actor(state).cpu().numpy().squeeze()

        if add_noise and self.training_mode:
            noise = self.noise.sample() * self.noise_scale
            action = action + noise
            # 限制动作范围
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def update_critic(self, batch, next_global_actions):
        """更新Critic网络"""
        self.critic.train()

        global_states = batch['global_states']
        global_actions = batch['global_actions']
        rewards = batch['agent_rewards'][:, self.agent_id].unsqueeze(1)
        next_global_states = batch['next_global_states']
        dones = batch['agent_dones'][:, self.agent_id].unsqueeze(1)

        # 计算目标Q值
        with torch.no_grad():
            target_q = self.critic_target(next_global_states, next_global_actions)
            target_q = rewards + (self.config.GAMMA * target_q * (1 - dones))

        # 当前Q值
        current_q = self.critic(global_states, global_actions)

        # Critic损失
        critic_loss = self.criterion(current_q, target_q)

        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        return critic_loss.item()

    def update_actor(self, batch, agents):
        """更新Actor网络"""
        self.actor.train()

        agent_states = batch['agent_states']
        global_states = batch['global_states']

        # 构建当前的全局动作
        actions = []
        for i, agent in enumerate(agents):
            if i == self.agent_id:
                # 使用当前智能体的Actor
                action = self.actor(agent_states[:, i])
            else:
                # 使用其他智能体的Actor (detach防止梯度传播)
                with torch.no_grad():
                    action = agent.actor(agent_states[:, i])
            actions.append(action)

        global_actions = torch.cat(actions, dim=1)

        # Actor损失：最大化Q值
        actor_loss = -self.critic(global_states, global_actions).mean()

        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        return actor_loss.item()

    def update_target_networks(self):
        """软更新目标网络"""
        soft_update(self.actor_target, self.actor, self.config.TAU)
        soft_update(self.critic_target, self.critic, self.config.TAU)

    def decay_noise(self):
        """衰减探索噪声"""
        self.noise_scale = max(self.noise_scale * self.noise_decay, self.min_noise)

    def reset_noise(self):
        """重置噪声"""
        self.noise.reset()

    def set_training_mode(self, mode=True):
        """设置训练模式"""
        self.training_mode = mode
        if mode:
            self.actor.train()
            self.critic.train()
        else:
            self.actor.eval()
            self.critic.eval()

    def save_models(self, filepath):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'noise_scale': self.noise_scale
        }, filepath)

    def load_models(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.noise_scale = checkpoint.get('noise_scale', self.config.NOISE_SCALE)

        # 更新目标网络
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)