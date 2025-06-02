import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agents.base_agent import BaseAgent

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Policy and Value networks
class ActorCriticNet(nn.Module):
    def __init__(self, input_dim=3, n_actions=4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        x = self.shared(state)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value

class PPOAgent(BaseAgent):
    def __init__(self, lr=3e-4, gamma=0.99, clip_eps=0.2, gae_lambda=0.95, k_epochs=4, batch_size=32):
        super().__init__()
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.k_epochs = k_epochs
        self.batch_size = batch_size

        self.policy = ActorCriticNet().to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.reset_episode_memory()

    def reset_episode_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def take_action(self, state: tuple[int, int]) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        logits, value = self.policy(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.states.append(state_tensor.squeeze(0))
        self.actions.append(action.item())
        self.log_probs.append(log_prob.item())
        self.values.append(value.item())

        return action.item()

    def update(self, next_state: tuple[int, int], reward: float, action_taken: int, done: bool = False):
        self.rewards.append(reward)
        self.dones.append(done)

        # If the episode is not done, skip the update
        if not done:
            return

        # Episode ended → time to update the networks
        self._ppo_update()
        self.reset_episode_memory()

    def _ppo_update(self):
        # Convert memory to tensors
        states = torch.stack(self.states).to(device)
        actions = torch.tensor(self.actions).to(device)
        rewards = torch.tensor(self.rewards).to(device)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(device)
        old_log_probs = torch.tensor(self.log_probs).to(device)
        values = torch.tensor(self.values).to(device)

        # Compute advantages and returns
        returns, advantages = self._compute_gae(rewards, values, dones)

        # Perform K PPO epochs
        for _ in range(self.k_epochs):
            logits, new_values = self.policy(states)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)

            # Ratio for clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

            # Loss = min(policy loss) - value loss
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(new_values.squeeze(), returns)
            loss = policy_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = values.tolist() + [0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = torch.tensor(advantages) + torch.tensor(values[:-1])
        return returns.to(device), torch.tensor(advantages).to(device)
