"""
https://github.com/higgsfield/RL-Adventure-2/blob/master/7.soft%20actor-critic.ipynb
"""
from collections import deque
import random

import numpy as np
import numpy.linalg as nla
import scipy.optimize as sop

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.distributions import Normal

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_size, init_w=3e-3):
        super().__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, init_w=3e-3):
        super().__init__()

        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, init_w=3e-3,
                 log_std_min=-20, log_std_max=2):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, action_size)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, action_size)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.Tensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.sigmoid(z)

        action = action.detach().cpu().numpy().ravel()
        return action


class SAC:
    def __init__(self, env, spec):
        self.batch_size = spec['agent']['sac']['batch_size']

        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        hidden_size = spec['agent']['sac']['hidden_size']

        self.state_size = state_size
        self.action_size = action_size

        self.replay_buffer = ReplayBuffer(maxlen=spec['agent']['sac']['buffer_size'])

        # Networks
        self.value_net = ValueNetwork(state_size, hidden_size)
        self.target_value_net = ValueNetwork(state_size, hidden_size)
        self.soft_q_net = SoftQNetwork(state_size, action_size, hidden_size)
        self.policy_net = PolicyNetwork(state_size, action_size, hidden_size)

        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()

        self.value_optimizer = optim.Adam(
            self.value_net.parameters(),
            lr=spec['agent']['sac']['value_lr']
        )
        self.soft_q_optimizer = optim.Adam(
            self.soft_q_net.parameters(),
            lr=spec['agent']['sac']['soft_q_lr']
        )
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=spec['agent']['sac']['policy_lr']
        )

    def act(self, obs):
        if len(obs) == self.state_size:
            dist = self.policy_net.get_action(obs) / self.action_size
        else:
            dist = np.zeros(self.action_size)

        return dist

    def soft_q_update(self, batch_size, gamma=0.99, mean_lambda=1e-3,
                      std_lambda=1e-3, z_lambda=0.0, soft_tau=1e-2):
        batch = self.replay_buffer.sample(batch_size)
        state, action, reward, next_state, done = batch

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        expected_q_value = self.soft_q_net(state, action)
        expected_value = self.value_net(state)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)

        target_value = self.target_value_net(next_state)
        next_q_value = reward + (1 - done) * gamma * target_value
        q_value_loss = self.soft_q_criterion(
            expected_q_value,
            next_q_value.detach())

        expected_new_q_value = self.soft_q_net(state, new_action)
        next_value = expected_new_q_value - log_prob
        value_loss = self.value_criterion(expected_value, next_value.detach())

        log_prob_target = expected_new_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

        mean_loss = mean_lambda * mean.pow(2).mean()
        std_loss = std_lambda * log_std.pow(2).mean()
        z_loss = z_lambda * z.pow(2).sum(1).mean()

        policy_loss += mean_loss + std_loss + z_loss

        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward()
        self.soft_q_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(),
                                       self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    def update(self, state, action, reward, next_state, done):
        if len(state) == self.state_size:
            self.replay_buffer.push(state, action, reward, next_state, done)

        if len(self.replay_buffer) > self.batch_size:
            self.soft_q_update(self.batch_size)


class ReplayBuffer:
    def __init__(self, maxlen):
        self.memory = deque(maxlen=int(maxlen))

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        unpacked = list(map(np.stack, zip(*batch)))
        return unpacked

    def __len__(self):
        return len(self.memory)


class NullAgent:
    def __init__(self, env, spec):
        self.action_shape = env.action_space.shape

    def act(self, obs):
        return np.zeros(self.action_shape)


if __name__ == '__main__':
    policy_network = PolicyNetwork(4, 3, 32)
