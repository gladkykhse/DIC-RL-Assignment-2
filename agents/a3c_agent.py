import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
import random
from agents.base_agent import BaseAgent

device = torch.device("cpu")  # A3C uses CPU for multiprocessing

class ActorCriticNet(nn.Module):
    def __init__(self, input_dim=3, n_actions=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
        )
        self.policy = nn.Linear(128, n_actions)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.policy(x), self.value(x)


class A3CAgent(BaseAgent):
    def __init__(self, global_model: nn.Module, optimizer: torch.optim.Optimizer, gamma=0.99):
        super().__init__()
        self.global_model = global_model
        self.optimizer = optimizer
        self.local_model = ActorCriticNet().to(device)
        self.local_model.load_state_dict(global_model.state_dict())
        self.gamma = gamma
        self.reset_memory()

    def reset_memory(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []

    def take_action(self, state: tuple[int, int, int]) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits, value = self.local_model(state_tensor)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        self.log_probs.append(dist.log_prob(action))
        self.values.append(value)
        self.entropies.append(dist.entropy())
        return action.item()

    def update(self, next_state, reward, action, done=False):
        self.rewards.append(reward)

        if not done:
            return

        R = 0 if done else self.local_model(torch.FloatTensor(next_state).unsqueeze(0))[1].item()
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)

        log_probs = torch.stack(self.log_probs)
        values = torch.cat(self.values).squeeze()
        entropies = torch.stack(self.entropies)

        advantage = returns - values

        policy_loss = -(log_probs * advantage.detach()).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()

        total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        for local_param, global_param in zip(self.local_model.parameters(), self.global_model.parameters()):
            global_param._grad = local_param.grad
        self.optimizer.step()
        self.local_model.load_state_dict(self.global_model.state_dict())

        self.reset_memory()
