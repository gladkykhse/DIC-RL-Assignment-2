import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from agents.base_agent import BaseAgent

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Policy and Value networks
class ActorCriticNet(nn.Module):
    def __init__(self, input_dim, n_actions):
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
    def __init__(self, input_dim=3, n_actions=4, lr=3e-4, gamma=0.99, clip_eps=0.2,
                 gae_lambda=0.95, k_epochs=4, batch_size=32,
                 value_loss_coef=0.5, entropy_coef=0.01):
        """
        Initializes the PPO Agent.

        Args:
            input_dim (int): Dimension of the observation space.
            n_actions (int): Number of possible actions.
            lr (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            clip_eps (float): Clipping parameter for the PPO objective.
            gae_lambda (float): Lambda parameter for Generalized Advantage Estimation.
            k_epochs (int): Number of epochs to run on the collected data for each update.
            batch_size (int): Size of minibatches for the PPO update.
            value_loss_coef (float): Coefficient for the value function loss in the total loss.
            entropy_coef (float): Coefficient for the entropy bonus in the policy loss (encourages exploration).
        """
        super().__init__()
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.grid_width = 1
        self.grid_height = 1
        self.max_targets = 1


        self.policy = ActorCriticNet(input_dim, n_actions).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.reset_episode_memory()

    def normalize_state(self, state: tuple) -> torch.Tensor:
        x, y, targets = state
        norm_x = x / self.grid_width
        norm_y = y / self.grid_height
        norm_targets = targets / self.max_targets if self.max_targets > 0 else 0
        return torch.FloatTensor([x, y, targets]).to(device)


    def reset_episode_memory(self):
        """
        Clears the stored episode data. This should be called after each episode update.
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def take_action(self, state: tuple) -> int:
        """
        Selects an action based on the current state using the policy network.
        Stores the state, action, log probability, and value for later updates.

        Args:
            state (tuple): The current state of the environment.

        Returns:
            int: The chosen action.
        """
        # Convert state tuple to a FloatTensor and add a batch dimension
        state_tensor = self.normalize_state(state).unsqueeze(0)

        with torch.no_grad(): # Actions are chosen without computing gradients
            logits, value = self.policy(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        # Store data for the PPO update
        self.states.append(state_tensor.squeeze(0)) # Remove batch dim before storing
        self.actions.append(action.item())
        self.log_probs.append(log_prob.item())
        self.values.append(value.item())

        return action.item()

    def update(self, next_state: tuple, reward: float, action_taken: int, done: bool = False):
        """
        Receives feedback from the environment and triggers a policy update if the episode is done.

        Args:
            next_state (tuple): The state after taking the action (not used in this update method,
                                but standard for RL agent interfaces).
            reward (float): The reward received from the environment.
            action_taken (int): The action that was taken (not used in this update method).
            done (bool): True if the episode has ended, False otherwise.
        """
        self.rewards.append(reward)
        self.dones.append(done)

        # If the episode is not done, we continue collecting data
        if not done:
            return

        # Episode ended → time to update the networks
        self._ppo_update()
        self.reset_episode_memory()

    def _ppo_update(self):
        """
        Performs the PPO update using the collected episode data.
        This involves computing advantages, returns, and then iterating
        for K epochs over minibatches of data to update the policy and value networks.
        """
        # Convert collected lists to PyTorch tensors
        states = torch.stack(self.states).to(device)
        actions = torch.tensor(self.actions).to(device)
        rewards = torch.tensor(self.rewards).to(device)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(device)
        old_log_probs = torch.tensor(self.log_probs).to(device)
        values = torch.tensor(self.values).to(device)

        # Compute advantages and returns using GAE
        returns, advantages = self._compute_gae(rewards, values, dones)

        # Normalize advantages for more stable training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Create a TensorDataset and DataLoader for minibatching
        dataset = TensorDataset(states, actions, old_log_probs, returns, advantages)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Perform K PPO epochs over the collected data
        for _ in range(self.k_epochs):
            for batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages in dataloader:
                # Get current logits and values from the policy network
                logits, new_values = self.policy(batch_states)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean() # Calculate entropy for bonus

                # Calculate the ratio for the clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages

                # Policy Loss (clipped objective + entropy bonus)
                policy_loss = -torch.min(surr1, surr2).mean()
                if self.entropy_coef > 0:
                    policy_loss -= self.entropy_coef * entropy # Subtract entropy for maximization

                # Value Loss (Mean Squared Error)
                value_loss = nn.functional.mse_loss(new_values.view(-1), batch_returns.view(-1))

                # Total Loss
                loss = policy_loss + self.value_loss_coef * value_loss

                # Optimize the network
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def _compute_gae(self, rewards, values, dones):
        """
        Computes Generalized Advantage Estimation (GAE) and discounted returns.

        Args:
            rewards (torch.Tensor): Tensor of rewards for the episode.
            values (torch.Tensor): Tensor of state values predicted by the critic.
            dones (torch.Tensor): Tensor indicating if a step was terminal (1) or not (0).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Returns (discounted returns, advantages).
        """
        advantages = []
        gae = 0
        # Append 0 to values to represent the value of the terminal state
        # Or, if the last state is not terminal, it could be the value of the next_state
        # For an episode that just finished (done=True), the last value is indeed 0.
        values_extended = values.tolist() + [0] # Convert to list for easier manipulation

        # Iterate backwards to compute GAE
        for t in reversed(range(len(rewards))):
            # Delta = R_t + gamma * V(S_{t+1}) * (1 - D_t) - V(S_t)
            delta = rewards[t] + self.gamma * values_extended[t + 1] * (1 - dones[t]) - values_extended[t]
            # GAE = delta + gamma * lambda * (1 - D_t) * GAE_{t+1}
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae) # Insert at the beginning to maintain chronological order

        advantages = torch.tensor(advantages).to(device)
        # Returns = Advantages + Values
        returns = advantages + values.to(device)

        return returns, advantages
