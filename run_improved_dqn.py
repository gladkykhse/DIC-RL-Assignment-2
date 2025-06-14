# Improved DQN Agent Runner
# Compatible with the improved training code

from argparse import ArgumentParser
from pathlib import Path
import numpy as np

try:
    from world import Environment
except ModuleNotFoundError:
    from os import path, pardir
    import sys
    root_path = path.abspath(path.join(path.abspath(__file__), pardir, pardir))
    if root_path not in sys.path:
        sys.path.append(root_path)
    from world import Environment

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# Add the ImprovedTrainingConfig class to avoid unpickling errors
class ImprovedTrainingConfig:
    """Dummy config class to handle model loading"""
    def __init__(self):
        self.state_dim = 8
        self.n_actions = 4
        self.hidden_size = 256
        self.buffer_capacity = 20000
        self.batch_size = 128
        self.gamma = 0.99
        self.lr = 5e-4
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay_steps = 5000
        self.target_update_freq = 200
        self.num_episodes = 1000
        self.max_steps_per_episode = 500
        self.warmup_episodes = 100
        self.lr_decay_steps = 1000
        self.lr_decay_rate = 0.95
        self.log_interval = 50
        self.save_interval = 300

class ImprovedDQN(nn.Module):
    """Improved DQN architecture matching the training code"""
    
    def __init__(self, input_dim: int, n_actions: int, hidden_size: int = 256):
        super().__init__()
        
        # Use LayerNorm instead of BatchNorm to handle single samples
        self.input_layer = nn.Linear(input_dim, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        
        # Hidden layers with residual connections
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.hidden3 = nn.Linear(hidden_size, hidden_size // 2)
        self.norm3 = nn.LayerNorm(hidden_size // 2)
        
        # Advantage and value streams (Dueling DQN)
        self.advantage = nn.Linear(hidden_size // 2, n_actions)
        self.value = nn.Linear(hidden_size // 2, 1)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input layer
        x = F.relu(self.input_norm(self.input_layer(x)))
        x = self.dropout(x)
        
        # Hidden layers with residual connections
        residual = x
        x = F.relu(self.norm1(self.hidden1(x)))
        x = self.dropout(x) + residual  # Residual connection
        
        residual = x
        x = F.relu(self.norm2(self.hidden2(x)))
        x = self.dropout(x) + residual  # Residual connection
        
        # Final hidden layer
        x = F.relu(self.norm3(self.hidden3(x)))
        x = self.dropout(x)
        
        # Dueling DQN: separate advantage and value streams
        advantage = self.advantage(x)
        value = self.value(x)
        
        # Combine advantage and value
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q_values

class ImprovedDQNAgent:
    """Improved DQN Agent compatible with the trained models"""
    
    def __init__(self, model_file, n_rows, n_cols, max_deliveries, device="cpu"):
        self.device = torch.device(device)
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.max_deliveries = max_deliveries
        
        # Enhanced state space size (matching improved training)
        self.state_dim = 8  # Enhanced features
        self.n_actions = 4
        self.hidden_size = 256
        
        # Create model with improved architecture
        self.model = ImprovedDQN(
            self.state_dim, 
            self.n_actions, 
            self.hidden_size
        ).to(self.device)
        
        # Load the saved weights with better error handling
        try:
            # First try loading with weights_only=False
            checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded checkpoint from episode {checkpoint.get('episode', 'unknown')}")
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Improved model loaded successfully from {model_file}")
        except Exception as e1:
            print(f"First attempt failed: {e1}")
            try:
                # Try loading only the model state dict
                checkpoint = torch.load(model_file, map_location=self.device, weights_only=True)
                self.model.load_state_dict(checkpoint)
                print(f"Loaded model weights only from {model_file}")
            except Exception as e2:
                print(f"Second attempt failed: {e2}")
                try:
                    # Last resort - try to extract just the model weights
                    import pickle
                    with open(model_file, 'rb') as f:
                        # Try to manually extract the model state dict
                        checkpoint = pickle.load(f)
                        if 'model_state_dict' in checkpoint:
                            self.model.load_state_dict(checkpoint['model_state_dict'])
                        else:
                            self.model.load_state_dict(checkpoint)
                    print(f"Loaded model using manual pickle extraction from {model_file}")
                except Exception as e3:
                    print(f"All loading attempts failed:")
                    print(f"  Attempt 1: {e1}")
                    print(f"  Attempt 2: {e2}")
                    print(f"  Attempt 3: {e3}")
                    print("\nTrying to create a compatible model file...")
                    
                    # Create a new model file with just the weights
                    try:
                        original = torch.load(model_file, map_location='cpu', weights_only=False)
                        if 'model_state_dict' in original:
                            new_file = model_file.replace('.pt', '_weights_only.pt')
                            torch.save(original['model_state_dict'], new_file)
                            print(f"Created weights-only file: {new_file}")
                            print("Please use this file for future runs or retrain the model.")
                            self.model.load_state_dict(original['model_state_dict'])
                        else:
                            raise Exception("Could not extract model state dict")
                    except Exception as e4:
                        print(f"Could not create compatible model file: {e4}")
                        raise Exception("Model loading failed completely. Please retrain the model.")
        
        self.model.eval()
        
        # Store a dummy grid for state encoding (will be updated)
        self.grid = None
    
    def encode_state_enhanced(self, raw_state):
        """Enhanced state encoding exactly as in training."""
        i, j, remaining = raw_state
        
        # Basic normalized position and remaining targets
        basic_features = [
            i / (self.n_rows - 1) if self.n_rows > 1 else 0,
            j / (self.n_cols - 1) if self.n_cols > 1 else 0,
            remaining / self.max_deliveries if self.max_deliveries > 0 else 0
        ]
        
        # Add spatial context features (wall distances)
        wall_distances = []
        if self.grid is not None:
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # right, left, down, up
                distance = 0
                ni, nj = i, j
                while (0 <= ni < self.n_rows and 0 <= nj < self.n_cols and 
                       self.grid[ni, nj] != 1):
                    distance += 1
                    ni += di
                    nj += dj
                wall_distances.append(distance / max(self.n_rows, self.n_cols))
        else:
            wall_distances = [0.5, 0.5, 0.5, 0.5]  # Default values
        
        # Add target information
        if remaining > 0 and self.grid is not None:
            target_positions = np.where(self.grid == 3)
            if len(target_positions[0]) > 0:
                # Distance to nearest target
                min_target_dist = float('inf')
                for ti, tj in zip(target_positions[0], target_positions[1]):
                    dist = abs(i - ti) + abs(j - tj)  # Manhattan distance
                    min_target_dist = min(min_target_dist, dist)
                basic_features.append(min_target_dist / (self.n_rows + self.n_cols))
            else:
                basic_features.append(0.0)
        else:
            # Distance to start position (assuming (0,0) is start)
            start_dist = (i + j) / (self.n_rows + self.n_cols) if (self.n_rows + self.n_cols) > 0 else 0
            basic_features.append(start_dist)
        
        # Combine all features
        all_features = basic_features + wall_distances
        
        return torch.tensor(all_features, device=self.device, dtype=torch.float32)
    
    def get_action(self, raw_state):
        """Get action from the improved model"""
        with torch.no_grad():
            state_tensor = self.encode_state_enhanced(raw_state)
            q_values = self.model(state_tensor.unsqueeze(0))
            return q_values.argmax(dim=1).item()
    
    def act(self, observation):
        """Interface method for the environment"""
        return self.get_action(observation)
    
    def take_action(self, state):
        # This method should match the interface expected by the environment
        # The state format from environment.step() is (row, col, deliveries_remaining)
        if isinstance(state, (tuple, list)) and len(state) == 3:
            return self.get_action(state)
        else:
            print(f"Warning: Unexpected state format in take_action: {type(state)}")
            print(f"State: {state}")
            return self.get_action(state)  # Try anyway

def parse_args():
    p = ArgumentParser(description="Improved DQN Agent Runner.")
    p.add_argument("GRID", type=Path, nargs="+", help="Paths to the grid file to use.")
    p.add_argument("--no_gui", action="store_true")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--iter", type=int, default=500)
    p.add_argument("--random_seed", type=int, default=0)
    return p.parse_args()

# Mapping for improved models
IMPROVED_MAPPING = {
    "grid_configs/medium_grid_new_1.npy": {
        "model_file": "models/medium_grid_new_1_improved_policy.pt",
        "n_rows": 10,
        "n_cols": 10,
        "max_deliveries": 3,
    },
    "grid_configs/medium_grid_new_2.npy": {
        "model_file": "models/medium_grid_new_2_improved_policy.pt",
        "n_rows": 10,
        "n_cols": 10,
        "max_deliveries": 2,
    },
    "grid_configs/medium_grid_new_3.npy": {
        "model_file": "models/medium_grid_new_3_improved_policy.pt",
        "n_rows": 10,
        "n_cols": 10,
        "max_deliveries": 2,
    }

}

def main(grid_paths, no_gui, max_steps, fps, sigma, random_seed):
    device = get_device()
    print(f"Using device: {device}")
    
    for grid in grid_paths:
        print(f"\nRunning improved DQN agent on {grid}")
        
        # Create environment
        env = Environment(grid, no_gui, sigma=sigma, target_fps=fps, random_seed=random_seed)
        env.reset()
        
        str_path = str(grid)
        
        if str_path not in IMPROVED_MAPPING:
            print(f"No improved model configuration found for {str_path}")
            print(f"Available configurations: {list(IMPROVED_MAPPING.keys())}")
            continue
        
        # Create improved agent
        agent = ImprovedDQNAgent(
            model_file=IMPROVED_MAPPING[str_path]["model_file"],
            n_rows=IMPROVED_MAPPING[str_path]["n_rows"],
            n_cols=IMPROVED_MAPPING[str_path]["n_cols"],
            max_deliveries=IMPROVED_MAPPING[str_path]["max_deliveries"],
            device=str(device)
        )
        
        # Set the grid for enhanced state encoding
        agent.grid = env.grid.copy()
        
        # Evaluate the agent
        print(f"Evaluating agent for {max_steps} steps...")
        Environment.evaluate_agent(grid, agent, max_steps, sigma, random_seed=random_seed)

if __name__ == "__main__":
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed)