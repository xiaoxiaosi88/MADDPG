import numpy as np
import random
from collections import deque, namedtuple
import torch


class ReplayBuffer:
    """优化后的经验回放缓冲区"""

    def __init__(self, capacity, n_agents):
        self.capacity = capacity
        self.n_agents = n_agents
        self.buffer = deque(maxlen=capacity)

        # 定义经验元组
        self.Experience = namedtuple('Experience', [
            'states', 'actions', 'rewards', 'next_states', 'dones',
            'global_state', 'global_action', 'next_global_state'
        ])

    def push(self, states, actions, rewards, next_states, dones,
             global_state, global_action, next_global_state):
        """添加经验到缓冲区"""
        # 确保所有输入都是numpy数组
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        global_state = np.array(global_state, dtype=np.float32)
        global_action = np.array(global_action, dtype=np.float32)
        next_global_state = np.array(next_global_state, dtype=np.float32)

        experience = self.Experience(
            states, actions, rewards, next_states, dones,
            global_state, global_action, next_global_state
        )
        self.buffer.append(experience)

    def sample(self, batch_size, device='cpu'):
        """采样一批经验 - 优化版本"""
        experiences = random.sample(self.buffer, batch_size)

        # 预分配numpy数组以提高效率
        batch_agent_states = np.zeros((batch_size, self.n_agents, experiences[0].states.shape[1]), dtype=np.float32)
        batch_agent_actions = np.zeros((batch_size, self.n_agents, experiences[0].actions.shape[1]), dtype=np.float32)
        batch_agent_rewards = np.zeros((batch_size, self.n_agents), dtype=np.float32)
        batch_agent_next_states = np.zeros((batch_size, self.n_agents, experiences[0].next_states.shape[1]),
                                           dtype=np.float32)
        batch_agent_dones = np.zeros((batch_size, self.n_agents), dtype=np.float32)

        batch_global_states = np.zeros((batch_size, experiences[0].global_state.shape[0]), dtype=np.float32)
        batch_global_actions = np.zeros((batch_size, experiences[0].global_action.shape[0]), dtype=np.float32)
        batch_next_global_states = np.zeros((batch_size, experiences[0].next_global_state.shape[0]), dtype=np.float32)

        # 填充数据
        for i, exp in enumerate(experiences):
            batch_agent_states[i] = exp.states
            batch_agent_actions[i] = exp.actions
            batch_agent_rewards[i] = exp.rewards
            batch_agent_next_states[i] = exp.next_states
            batch_agent_dones[i] = exp.dones
            batch_global_states[i] = exp.global_state
            batch_global_actions[i] = exp.global_action
            batch_next_global_states[i] = exp.next_global_state

        # 一次性转换为tensor（更高效）
        return {
            'agent_states': torch.from_numpy(batch_agent_states).to(device),
            'agent_actions': torch.from_numpy(batch_agent_actions).to(device),
            'agent_rewards': torch.from_numpy(batch_agent_rewards).to(device),
            'agent_next_states': torch.from_numpy(batch_agent_next_states).to(device),
            'agent_dones': torch.from_numpy(batch_agent_dones).to(device),
            'global_states': torch.from_numpy(batch_global_states).to(device),
            'global_actions': torch.from_numpy(batch_global_actions).to(device),
            'next_global_states': torch.from_numpy(batch_next_global_states).to(device)
        }

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """优化后的优先级经验回放缓冲区"""

    def __init__(self, capacity, n_agents, alpha=0.6):
        self.capacity = capacity
        self.n_agents = n_agents
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

        self.Experience = namedtuple('Experience', [
            'states', 'actions', 'rewards', 'next_states', 'dones',
            'global_state', 'global_action', 'next_global_state'
        ])

    def push(self, states, actions, rewards, next_states, dones,
             global_state, global_action, next_global_state):
        """添加经验到缓冲区"""
        # 确保所有输入都是numpy数组
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        global_state = np.array(global_state, dtype=np.float32)
        global_action = np.array(global_action, dtype=np.float32)
        next_global_state = np.array(next_global_state, dtype=np.float32)

        experience = self.Experience(
            states, actions, rewards, next_states, dones,
            global_state, global_action, next_global_state
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience

        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4, device='cpu'):
        """根据优先级采样经验"""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]

        # 计算重要性权重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        # 预分配numpy数组以提高效率
        batch_agent_states = np.zeros((batch_size, self.n_agents, experiences[0].states.shape[1]), dtype=np.float32)
        batch_agent_actions = np.zeros((batch_size, self.n_agents, experiences[0].actions.shape[1]), dtype=np.float32)
        batch_agent_rewards = np.zeros((batch_size, self.n_agents), dtype=np.float32)
        batch_agent_next_states = np.zeros((batch_size, self.n_agents, experiences[0].next_states.shape[1]),
                                           dtype=np.float32)
        batch_agent_dones = np.zeros((batch_size, self.n_agents), dtype=np.float32)

        batch_global_states = np.zeros((batch_size, experiences[0].global_state.shape[0]), dtype=np.float32)
        batch_global_actions = np.zeros((batch_size, experiences[0].global_action.shape[0]), dtype=np.float32)
        batch_next_global_states = np.zeros((batch_size, experiences[0].next_global_state.shape[0]), dtype=np.float32)

        # 填充数据
        for i, exp in enumerate(experiences):
            batch_agent_states[i] = exp.states
            batch_agent_actions[i] = exp.actions
            batch_agent_rewards[i] = exp.rewards
            batch_agent_next_states[i] = exp.next_states
            batch_agent_dones[i] = exp.dones
            batch_global_states[i] = exp.global_state
            batch_global_actions[i] = exp.global_action
            batch_next_global_states[i] = exp.next_global_state

        return {
            'agent_states': torch.from_numpy(batch_agent_states).to(device),
            'agent_actions': torch.from_numpy(batch_agent_actions).to(device),
            'agent_rewards': torch.from_numpy(batch_agent_rewards).to(device),
            'agent_next_states': torch.from_numpy(batch_agent_next_states).to(device),
            'agent_dones': torch.from_numpy(batch_agent_dones).to(device),
            'global_states': torch.from_numpy(batch_global_states).to(device),
            'global_actions': torch.from_numpy(batch_global_actions).to(device),
            'next_global_states': torch.from_numpy(batch_next_global_states).to(device),
            'weights': torch.from_numpy(weights).to(device),
            'indices': indices
        }

    def update_priorities(self, indices, priorities):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)