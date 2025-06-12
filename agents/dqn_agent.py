# agents/dqn_agent.py

import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent

# A simple namedtuple to store transitions
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        """Randomly sample a batch of transitions"""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent(BaseAgent):
    def __init__(
        self,
        state_dim: int = 3,
        n_actions: int = 4,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 500,
    ):
        """
        Deep Q-Network agent.

        Args:
            state_dim: Dimension of the continuous state vector (e.g. 3).
            n_actions: Number of discrete actions.
            hidden_dim: Number of hidden units in each MLP layer.
            lr: Learning rate for the optimizer.
            gamma: Discount factor.
            epsilon_start: Initial epsilon for epsilon-greedy.
            epsilon_end: Final epsilon after decay.
            epsilon_decay: Multiplicative decay factor per update.
            buffer_capacity: Maximum number of transitions in replay buffer.
            batch_size: Batch size for sampling from replay buffer.
            target_update_freq: How often (in gradient steps) to copy policy_net to target_net.
        """
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Policy (online) network and target network
        self.policy_net = QNetwork(input_dim=state_dim, hidden_dim=hidden_dim, n_actions=n_actions).to(self.device)
        self.target_net = QNetwork(input_dim=state_dim, hidden_dim=hidden_dim, n_actions=n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.steps_done = 0

    def take_action(self, state: np.ndarray) -> int:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: A 1D NumPy array of floats (e.g. [x_norm, y_norm, r_norm]).
        Returns:
            An integer action in [0, n_actions).
        """
        # Convert state to torch tensor
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)  # shape: [1, state_dim]
        sample = random.random()
        if sample < self.epsilon:
            # Explore: random action
            return random.randrange(self.n_actions)
        else:
            # Exploit: pick action with highest Q-value
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)  # shape: [1, n_actions]
                return int(torch.argmax(q_values, dim=1).item())

    def update(
        self,
        prev_state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        next_state: np.ndarray
    ):
        """
        Store the transition in replay buffer and perform a learning step if enough samples exist.

        Args:
            prev_state: 1D NumPy array of floats (previous state).
            action: The integer action taken.
            reward: The float reward received.
            done: Boolean flag indicating if the episode ended after this step.
            next_state: 1D NumPy array of floats (next state).
        """
        # 1. Store transition
        self.replay_buffer.push(
            np.array(prev_state, dtype=np.float32),
            action,
            float(reward),
            np.array(next_state, dtype=np.float32),
            done
        )

        # 2. Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # 3. Only learn if we have enough transitions
        if len(self.replay_buffer) < self.batch_size:
            return

        # 4. Sample a batch
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # 5. Convert batch elements to torch tensors
        state_batch = torch.from_numpy(np.vstack(batch.state)).float().to(self.device)            # [batch_size, state_dim]
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)  # [batch_size, 1]
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1) # [batch_size, 1]
        next_state_batch = torch.from_numpy(np.vstack(batch.next_state)).float().to(self.device)   # [batch_size, state_dim]
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)    # [batch_size, 1]

        # 6. Compute Q(s,a) using the policy network
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)  # [batch_size, 1]

        # 7. Compute target Q-values using the target network
        with torch.no_grad():
            max_next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)  # [batch_size, 1]
            target_q_values = reward_batch + (self.gamma * max_next_q_values * (1.0 - done_batch))

        # 8. Compute loss (MSE between current Q and target Q)
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # 9. Backpropagate and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 10. Periodically update the target network
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
