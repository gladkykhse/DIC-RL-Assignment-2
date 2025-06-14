from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange

try:
    from world import Environment
    from agents.dqn_agent import DQNAgent
except ModuleNotFoundError:
    from os import path, pardir
    import sys

    root_path = path.abspath(path.join(path.abspath(__file__), pardir, pardir))
    if root_path not in sys.path:
        sys.path.append(root_path)
    from world import Environment
    from agents.q_learning import QLearningAgent

# Import the compatible agent
import torch
import torch.nn as nn

class TrainedDQN(nn.Module):
    """DQN architecture matching the training code"""
    
    def __init__(self, input_dim: int, n_actions: int, hidden_size: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),           # network.0
            nn.ReLU(),                                   # network.1
            nn.Dropout(0.1),                             # network.2
            nn.Linear(hidden_size, hidden_size),         # network.3
            nn.ReLU(),                                   # network.4
            nn.Dropout(0.1),                             # network.5
            nn.Linear(hidden_size, hidden_size // 2),    # network.6
            nn.ReLU(),                                   # network.7
            nn.Linear(hidden_size // 2, n_actions)       # network.8
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CompatibleDQNAgent:
    """DQN Agent compatible with the trained models"""
    
    def __init__(self, model_file, n_rows, n_cols, max_deliveries, device="cpu"):
        self.device = device
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.max_deliveries = max_deliveries
        
        # State space size (matching training: row, col, deliveries_remaining)
        self.state_dim = 3
        self.n_actions = 4
        self.hidden_size = 128
        
        # Create model with exact same architecture as training
        self.model = TrainedDQN(
            self.state_dim, 
            self.n_actions, 
            self.hidden_size
        ).to(self.device)
        
        # Load the saved weights
        try:
            state_dict = torch.load(model_file, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Model loaded successfully from {model_file}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
        
        self.model.eval()
    
    def encode_state_normalized(self, raw_state):
        """Encode state exactly as in training (normalized)"""
        i, j, remaining = raw_state
        return torch.tensor([
            i / (self.n_rows - 1),
            j / (self.n_cols - 1), 
            remaining / self.max_deliveries
        ], device=self.device, dtype=torch.float32)
    
    def get_action(self, raw_state):
        """Get action from the model"""
        with torch.no_grad():
            state_tensor = self.encode_state_normalized(raw_state)
            q_values = self.model(state_tensor.unsqueeze(0))
            return q_values.argmax(dim=1).item()
    
    def act(self, observation):
        """Interface method for the environment"""
        # Handle different observation formats
        if isinstance(observation, dict):
            # Extract state from dict format
            agent_pos = observation.get('agent_positions', [None])[0]
            if agent_pos is None:
                print("Warning: Could not extract agent position from observation")
                return 0  # Default action
            
            deliveries_remaining = observation.get('deliveries_remaining', self.max_deliveries)
            raw_state = (agent_pos[0], agent_pos[1], deliveries_remaining)
        elif isinstance(observation, (tuple, list)) and len(observation) == 3:
            # Direct state format
            raw_state = observation
        else:
            print(f"Warning: Unexpected observation format: {type(observation)}")
            print(f"Observation: {observation}")
            return 0  # Default action
        
        return self.get_action(raw_state)
    
    def take_action(self, state):
        """Interface method expected by evaluate_agent"""
        # This method should match the interface expected by the environment
        # The state format from environment.step() is (row, col, deliveries_remaining)
        if isinstance(state, (tuple, list)) and len(state) == 3:
            return self.get_action(state)
        else:
            print(f"Warning: Unexpected state format in take_action: {type(state)}")
            print(f"State: {state}")
            return self.get_action(state)  # Try anyway

def parse_args():
    p = ArgumentParser(description="DIC Q-Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+", help="Paths to the grid file to use.")
    p.add_argument("--no_gui", action="store_true")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--iter", type=int, default=500)
    p.add_argument("--random_seed", type=int, default=0)
    return p.parse_args()


MAPPING = {
    "grid_configs/small_grid.npy": {
        "model_file": "models/small_grid_policy.pt",
        "n_rows": 8,
        "n_cols": 8,
        "max_deliveries": 1,
    },
    "grid_configs/small_grid_2.npy": {
        "model_file": "models/small_grid_2_policy.pt",
        "n_rows": 8,
        "n_cols": 8,
        "max_deliveries": 2,
    },
    "grid_configs/A1_grid.npy": {
        "model_file": "models/A1_grid_policy.pt",
        "n_rows": 15,
        "n_cols": 15,
        "max_deliveries": 1,
    },
    "grid_configs/medium_grid_new_1.npy": {
        "model_file": "models/medium_grid_new_1_policy.pt",
        "n_rows": 10,
        "n_cols": 10,
        "max_deliveries": 3,
    },
    "grid_configs/medium_grid_new_2.npy": {
        "model_file": "models/medium_grid_new_2_policy.pt",
        "n_rows": 10,
        "n_cols": 10,
        "max_deliveries": 2,
    }
}


def main(grid_paths, no_gui, max_steps, fps, sigma, random_seed):
    for grid in grid_paths:
        env = Environment(grid, no_gui, sigma=sigma, target_fps=fps, random_seed=random_seed)
        env.reset()

        str_path = str(grid)
        config = MAPPING[str_path]
        
        print(f"Loading model: {config['model_file']}")
        print(f"Grid config: {config['n_rows']}x{config['n_cols']}, max_deliveries: {config['max_deliveries']}")
        
        # Use the compatible agent
        agent = CompatibleDQNAgent(
            model_file=config["model_file"],
            n_rows=config["n_rows"],
            n_cols=config["n_cols"],
            max_deliveries=config["max_deliveries"],
            device="cpu"
        )

        print("Starting evaluation...")
        Environment.evaluate_agent(grid, agent, max_steps, sigma, random_seed=random_seed)


if __name__ == "__main__":
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed)