import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agents import BaseAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        shared = self.shared(x)
        logits = self.actor(shared)
        value = self.critic(shared)
        return logits, value

class PPOAgent(BaseAgent):
    def __init__(self, state_dim=2, action_dim=4, lr=3e-4, gamma=0.99, k_epochs=4, gae_lambda=0.95, batch_size=32, entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5, clip_eps=0.2):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.clip_eps = clip_eps

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def reset_buffer(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def take_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        logits, value = self.policy(state_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        self.states.append(state_tensor.squeeze(0))
        self.log_probs.append(dist.log_prob(action))
        self.actions.append(action)
        self.values.append(value)
        return action.item()

    def update(self, next_state, reward, action_taken, done=False):
        self.rewards.append(reward)
        self.dones.append(done)

        if done:
            self._ppo_update()
            self.reset_buffer()

    def _ppo_update(self):
        # Stack episode data
        states = torch.stack(self.states).to(device)
        actions = torch.stack(self.actions).to(device)
        old_log_probs = torch.stack(self.log_probs).to(device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)
        # values = torch.cat(self.values + [torch.tensor([0.], device=device)])  
        values = torch.cat([v.squeeze(-1) for v in self.values] + [torch.tensor([0.], device=device)])

        # Compute GAE advantages and returns
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, device=device)
        returns = advantages + values[:-1]

        # PPO update loop
        N = len(states)
        for _ in range(self.k_epochs):
            perm = torch.randperm(N)
            for i in range(0, N, self.batch_size):
                idx = perm[i:i + self.batch_size]
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                logits, values = self.policy(batch_states)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO clipped loss
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, batch_returns)

                # Final loss and update
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
