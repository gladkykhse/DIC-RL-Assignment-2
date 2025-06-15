import numpy as np
import torch.nn as nn
import torch

from agents import BaseAgent


class DQN(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden_size: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DQNAgent(BaseAgent):
    def __init__(
        self,
        model_file: str,
        hidden_size: int,
        device: torch.device,
        n_rows: int,
        n_cols: int,
        max_deliveries: int,
    ):
        super().__init__()
        self.device = device
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.max_deliveries = max_deliveries

        self.model = DQN(input_dim=5, n_actions=4, hidden_size=hidden_size).to(device)
        self.model.load_state_dict(torch.load(model_file, map_location=device))
        self.model.eval()

    def encode_state_norm(self, raw: tuple[int, int, int, int, int]) -> torch.Tensor:
        start_x, start_y, agent_x, agent_y, rem = raw
        return torch.tensor(
            [start_x / (self.n_rows - 1), start_y / (self.n_cols - 1), 
             agent_x / (self.n_rows - 1), agent_y / (self.n_cols - 1), 
             rem / self.max_deliveries],
            device=self.device,
            dtype=torch.float32,
        )

    def update(self, state: tuple[int, int], reward: float, action):
        pass

    def take_action(self, state: tuple[int, int, int, int, int]) -> int:
        state_tensor = self.encode_state_norm(state)
        action = self.model(state_tensor.unsqueeze(0)).argmax(dim=1).item()
        return action
