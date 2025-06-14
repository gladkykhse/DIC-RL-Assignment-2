
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path
from tqdm import trange
import sys
import os

if os.path.abspath('.') not in sys.path:
    sys.path.append(os.path.abspath('.'))

from world.delivery_environment import Environment
from enhanced_reward_function import EnhancedRewardWrapper

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")



class ImprovedDQN(nn.Module):
    """Improved DQN with LayerNorm instead of BatchNorm to handle single samples."""
    
    def __init__(self, input_dim: int, n_actions: int, hidden_size: int = 256):
        super().__init__()
        
        
        self.input_layer = nn.Linear(input_dim, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        
       
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.hidden3 = nn.Linear(hidden_size, hidden_size // 2)
        self.norm3 = nn.LayerNorm(hidden_size // 2)
        
        
        self.advantage = nn.Linear(hidden_size // 2, n_actions)
        self.value = nn.Linear(hidden_size // 2, 1)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input layer
        x = F.relu(self.input_norm(self.input_layer(x)))
        x = self.dropout(x)
        
        
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


# ENHANCED STATE ENCODING


def encode_state_enhanced(raw_state: tuple, grid: np.ndarray, 
                         n_rows: int, n_cols: int, max_deliveries: int) -> torch.Tensor:
    """Enhanced state encoding with more spatial information."""
    i, j, remaining = raw_state
    
    # Basic normalized position and remaining targets
    basic_features = [
        i / (n_rows - 1),
        j / (n_cols - 1), 
        remaining / max_deliveries
    ]
    
    # Add spatial context features
    # Distance to nearest wall
    wall_distances = []
    for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]: 
        distance = 0
        ni, nj = i, j
        while 0 <= ni < n_rows and 0 <= nj < n_cols and grid[ni, nj] != 1:
            distance += 1
            ni += di
            nj += dj
        wall_distances.append(distance / max(n_rows, n_cols))
    
    # Add target information if any targets remain
    if remaining > 0:
        target_positions = np.where(grid == 3)
        if len(target_positions[0]) > 0:
            # Distance to nearest target
            min_target_dist = float('inf')
            for ti, tj in zip(target_positions[0], target_positions[1]):
                dist = abs(i - ti) + abs(j - tj)  # Manhattan distance
                min_target_dist = min(min_target_dist, dist)
            basic_features.append(min_target_dist / (n_rows + n_cols))
        else:
            basic_features.append(0.0)
    else:
        # Distance to start position (assuming (0,0) is start)
        start_dist = (i + j) / (n_rows + n_cols)
        basic_features.append(start_dist)
    
    # Combine all features
    all_features = basic_features + wall_distances
    
    return torch.tensor(all_features, device=device, dtype=torch.float32)


# IMPROVED REPLAY BUFFER WITH PRIORITIZED EXPERIENCE REPLAY


class ImprovedReplayBuffer:
    """Improved replay buffer with better prioritization."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4   # Importance sampling exponent
        self.beta_increment = 0.001
        self.max_priority = 1.0
        
    def push(self, state, action, reward, next_state, done, td_error=None):
        self.buffer.append((state, action, reward, next_state, done))
        
        # Set priority based on TD error or max priority for new experiences
        if td_error is not None:
            priority = (abs(td_error) + 1e-6) ** self.alpha
        else:
            priority = self.max_priority
            
        self.priorities.append(priority)
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int, grid: np.ndarray, n_rows: int, n_cols: int, max_deliveries: int):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        
        # Sample based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        batch = [self.buffer[i] for i in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Encode states with enhanced features
        states_enc = torch.stack([encode_state_enhanced(s, grid, n_rows, n_cols, max_deliveries) for s in states])
        next_states_enc = torch.stack([encode_state_enhanced(s, grid, n_rows, n_cols, max_deliveries) for s in next_states])
        
        # Update beta for importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return (
            states_enc,
            torch.tensor(actions, dtype=torch.int64, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            next_states_enc,
            torch.tensor(dones, dtype=torch.float32, device=device),
            torch.tensor(weights, dtype=torch.float32, device=device),
            indices
        )
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


# IMPROVED TRAINING CONFIGURATION


class ImprovedTrainingConfig:
    def __init__(self):
        # Network parameters
        self.state_dim = 8  
        self.n_actions = 4
        self.hidden_size = 256  
        
        # Training parameters
        self.buffer_capacity = 20000  
        self.batch_size = 128 
        self.gamma = 0.99
        self.lr = 5e-4  
        
        # Exploration parameters with better schedule
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay_steps = 5000 
        
        # Training schedule
        self.target_update_freq = 200
        self.num_episodes = 1000  
        self.max_steps_per_episode = 500 
        self.warmup_episodes = 100 
        
        # Learning rate scheduling
        self.lr_decay_steps = 1000
        self.lr_decay_rate = 0.95
        
        # Logging
        self.log_interval = 50
        self.save_interval = 300


# CURRICULUM LEARNING FUNCTION


def get_epsilon_for_episode(episode: int, config: ImprovedTrainingConfig) -> float:
    """Get epsilon value for current episode with improved scheduling."""
    if episode < config.warmup_episodes:
        return config.epsilon_start
    
    progress = min(1.0, (episode - config.warmup_episodes) / config.epsilon_decay_steps)
    epsilon = config.epsilon_start + (config.epsilon_end - config.epsilon_start) * progress
    
    return max(config.epsilon_end, epsilon)


# IMPROVED TRAINING FUNCTION


def train_improved_dqn(grid_name="medium_grid_new_3.npy", config=None):
    """Train improved DQN agent with better architecture and training."""
    
    if config is None:
        config = ImprovedTrainingConfig()
    
    # Setup environment
    from world.delivery_environment import Environment
    from enhanced_reward_function import EnhancedRewardWrapper
    
    grid_path = Path(f"grid_configs/{grid_name}")
    if not grid_path.exists():
        print(f"Grid {grid_path} not found.")
        return None
    
    base_env = Environment(grid_path, no_gui=True, random_seed=42)
    env = EnhancedRewardWrapper(base_env)
    
    
    env.reset()
    n_rows, n_cols = env.grid.shape
    max_deliveries = env.initial_target_count
    
    print(f"Training improved DQN on {grid_name}")
    print(f"Grid size: {n_rows}x{n_cols}")
    print(f"Max deliveries: {max_deliveries}")
    print(f"Enhanced state encoding with {config.state_dim} features")
    
    # Initialize networks
    policy_net = ImprovedDQN(config.state_dim, config.n_actions, config.hidden_size).to(device)
    target_net = ImprovedDQN(config.state_dim, config.n_actions, config.hidden_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # Initialize optimizer with learning rate scheduling
    optimizer = optim.Adam(policy_net.parameters(), lr=config.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_steps, gamma=config.lr_decay_rate)
    
    # Initialize improved replay buffer
    replay_buffer = ImprovedReplayBuffer(config.buffer_capacity)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    success_rate = []
    recent_successes = deque(maxlen=100)
    td_errors = []
    
    print("Starting improved training...")
    
    for episode in trange(1, config.num_episodes + 1, desc="Training"):
        raw_state = env.reset()
        total_reward = 0
        steps = 0
        episode_td_errors = []
        
        # Get epsilon for this episode
        epsilon = get_epsilon_for_episode(episode, config)
        
        for step in range(config.max_steps_per_episode):
            steps += 1
            
            # Enhanced epsilon-greedy with better exploration
            if random.random() < epsilon:
                action = random.randrange(config.n_actions)
            else:
                with torch.no_grad():
                    # Set model to eval mode for inference to handle dropout properly
                    policy_net.eval()
                    state_tensor = encode_state_enhanced(raw_state, env.grid, n_rows, n_cols, max_deliveries)
                    q_values = policy_net(state_tensor.unsqueeze(0))
                    action = q_values.argmax(dim=1).item()
                    # Set back to train mode
                    policy_net.train()
            
            # Take action
            raw_next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Store transition
            replay_buffer.push(raw_state, action, reward, raw_next_state, done)
            
            raw_state = raw_next_state
            
            # Training step (only after warmup)
            if episode > config.warmup_episodes and len(replay_buffer) >= config.batch_size:
                # Sample batch with importance sampling
                states, actions, rewards, next_states, dones, weights, indices = replay_buffer.sample(
                    config.batch_size, env.grid, n_rows, n_cols, max_deliveries
                )
                
                # Ensure network is in training mode
                policy_net.train()
                
                # Compute Q-values
                current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # Double DQN target computation
                with torch.no_grad():
                    # Use policy network to select actions
                    policy_net.eval()
                    next_actions = policy_net(next_states).argmax(dim=1)
                    # Use target network to evaluate actions
                    target_net.eval()
                    next_q_values = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    target_q_values = rewards + config.gamma * next_q_values * (1 - dones)
                    policy_net.train()
                
                # Compute TD errors for priority updates
                td_error = target_q_values - current_q_values
                episode_td_errors.extend(td_error.detach().cpu().numpy())
                
                # Weighted MSE loss for importance sampling
                loss = (weights * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
                
                # Update priorities in replay buffer
                replay_buffer.update_priorities(indices, td_error.detach().cpu().numpy())
            
            if done:
                break
        
        # Update learning rate
        if episode > config.warmup_episodes:
            scheduler.step()
        
        # Track success
        success = done and env.world_stats.get("final_bonus_given", 0) > 0
        recent_successes.append(success)
        
        # Update target network
        if episode % config.target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Store metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        success_rate.append(np.mean(recent_successes) if recent_successes else 0)
        if episode_td_errors:
            td_errors.append(np.mean(np.abs(episode_td_errors)))
        
        # Logging
        if episode % config.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-config.log_interval:])
            avg_length = np.mean(episode_lengths[-config.log_interval:])
            current_success_rate = success_rate[-1]
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Episode {episode:4d} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Avg Steps: {avg_length:5.1f} | "
                  f"Success Rate: {current_success_rate:.3f} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"LR: {current_lr:.6f}")
        
        # Save model periodically
        if episode % config.save_interval == 0:
            model_path = Path("models")
            model_path.mkdir(exist_ok=True)
            model_name = grid_name.replace('.npy', '_improved_policy.pt')
            torch.save({
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'episode': episode
            }, model_path / model_name)
    
    # Final save
    model_path = Path("models")
    model_path.mkdir(exist_ok=True)
    model_name = grid_name.replace('.npy', '_improved_policy.pt')
    torch.save({
        'model_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'episode': episode
    }, model_path / model_name)
    
    print(f"Improved training completed! Model saved as {model_name}")
    
    return {
        'policy_net': policy_net,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_rate': success_rate,
        'td_errors': td_errors,
        'config': config
    }


# IMPROVED EVALUATION FUNCTION


def evaluate_improved_agent(grid_name="medium_grid_new_3.npy", model_path=None, num_episodes=20):
    """Evaluate improved trained agent."""
    
    # Setup environment
    grid_path = Path(f"grid_configs/{grid_name}")
    base_env = Environment(grid_path, no_gui=True, random_seed=42)
    env = EnhancedRewardWrapper(base_env)
    
    n_rows, n_cols = env.grid.shape
    max_deliveries = env.initial_target_count
    
    # Load model
    if model_path is None:
        model_path = Path("models") / grid_name.replace('.npy', '_improved_policy.pt')
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', ImprovedTrainingConfig())
    
    policy_net = ImprovedDQN(config.state_dim, config.n_actions, config.hidden_size).to(device)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    policy_net.eval()
    
    print(f"Evaluating improved agent on {grid_name} for {num_episodes} episodes...")
    
    results = []
    
    for episode in range(num_episodes):
        raw_state = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(1000):
            steps += 1
            
            with torch.no_grad():
                state_tensor = encode_state_enhanced(raw_state, env.grid, n_rows, n_cols, max_deliveries)
                q_values = policy_net(state_tensor.unsqueeze(0))
                action = q_values.argmax(dim=1).item()
            
            raw_state, reward, done, info = env.step(action)
            total_reward += reward
            
            if done:
                break
        
        success = done and env.world_stats.get("final_bonus_given", 0) > 0
        results.append({
            'episode': episode + 1,
            'reward': total_reward,
            'steps': steps,
            'success': success,
            'targets_collected': env.world_stats.get("total_targets_reached", 0),
            'failed_moves': env.world_stats.get("total_failed_moves", 0)
        })
    
    # Print results
    successes = sum(r['success'] for r in results)
    avg_reward = np.mean([r['reward'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    avg_targets = np.mean([r['targets_collected'] for r in results])
    
    print(f"\nImproved Agent Evaluation Results:")
    print(f"Success Rate: {successes}/{num_episodes} ({successes/num_episodes*100:.1f}%)")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Average Targets Collected: {avg_targets:.1f}/{max_deliveries}")
    
    return results


# MAIN EXECUTION


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TRAINING IMPROVED DQN AGENT")
    print("="*60)
    
    config = ImprovedTrainingConfig()
    training_results = train_improved_dqn("medium_grid_new_3.npy", config)
    
    if training_results:
        print("\n" + "="*60)
        print("EVALUATING IMPROVED AGENT")
        print("="*60)
        
        evaluation_results = evaluate_improved_agent("medium_grid_new_3.npy")
        
        print("\nImproved training and evaluation completed!")