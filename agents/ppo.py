"""PPO Agent.

An agent that uses Proximal Policy Optimization with separate actor and critic networks.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from agents import BaseAgent

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
    

class PPOAgent(BaseAgent):
    def __init__(self,
                 state_dim=5,
                 action_dim=4,
                 hidden_dim=128,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_epsilon=0.2,
                 lr=3e-4,
                 update_epochs=4,
                 batch_size=64,
                 entropy_coef=0.01,
                 value_coef=0.5,
                 max_grad_norm=0.5):
        super().__init__()

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.policy_net = PolicyNetwork(state_dim, hidden_dim, action_dim).to(_DEVICE)
        self.value_net = ValueNetwork(state_dim, hidden_dim).to(_DEVICE)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        self.buffer = []

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=_DEVICE).unsqueeze(0)
        logits = self.policy_net(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item()

    def store_transition(self, state, action, reward, done, next_state):
        self.buffer.append((state, action, reward, done, next_state))

    def _compute_returns_and_advantages(self):
        states, actions, rewards, dones, next_states = zip(*self.buffer)

        states = torch.tensor(states, dtype=torch.float32, device=_DEVICE)
        actions = torch.tensor(actions, device=_DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=_DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32, device=_DEVICE)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=_DEVICE)

        values = self.value_net(states).detach()
        next_values = self.value_net(next_states).detach()

        returns, advantages = [], []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values[t] * mask - values[t]
            gae   = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        returns = torch.stack(returns)
        advantages = torch.stack(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return states, actions, returns, advantages

    def update(self):
        states, actions, returns, advantages = self._compute_returns_and_advantages()

        # freeze the probabilities of the behaviour policy
        with torch.no_grad():
            logits_old = self.policy_net(states)
            old_log_probs = torch.distributions.Categorical(logits=logits_old).log_prob(actions)

        N = states.size(0)
        for _ in range(self.update_epochs):
            # shuffle indices every epoch
            perm = torch.randperm(N, device=_DEVICE)

            for start in range(0, N, self.batch_size):
                idx = perm[start : start + self.batch_size]
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]
                batch_old_log_probs = old_log_probs[idx]
                
                # actor loss
                logits = self.policy_net(batch_states)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(batch_actions)

                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = dist.entropy().mean()

                # critic loss
                value_pred  = self.value_net(batch_states)
                value_loss  = nn.functional.mse_loss(value_pred, batch_returns)
                
                # optimazation
                self.policy_optimizer.zero_grad()
                (policy_loss - self.entropy_coef * entropy_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                (self.value_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.value_optimizer.step()

        self.buffer = []
