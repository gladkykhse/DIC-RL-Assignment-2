# Medium Grid DQN Training Notebook
# Optimized for faster experimentation with enhanced reward function

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

# Add the project root to path if needed
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

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def setup_environment(grid_name="medium_grid.npy"):
    """Setup environment with enhanced rewards."""
    grid_path = Path(f"grid_configs/{grid_name}")
    
    if not grid_path.exists():
        print(f"Grid {grid_path} not found. Please run the grid generator first.")
        return None
    
    # Create base environment
    base_env = Environment(grid_path, no_gui=True, random_seed=42)
    
    # Wrap with enhanced reward function
    env = EnhancedRewardWrapper(base_env)
    
    return env

# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================

class DQN(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden_size: int = 128):
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

# ============================================================================
# STATE ENCODING
# ============================================================================

def encode_state_normalized(raw_state: tuple, n_rows: int, n_cols: int, max_deliveries: int) -> torch.Tensor:
    """
    Encode state with normalization for better training stability.
    """
    i, j, remaining = raw_state
    return torch.tensor([
        i / (n_rows - 1),
        j / (n_cols - 1), 
        remaining / max_deliveries
    ], device=device, dtype=torch.float32)

# ============================================================================
# REPLAY BUFFER
# ============================================================================

class PrioritizedReplayBuffer:
    """Simple prioritized experience replay buffer."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = 0.6  # Priority exponent
        
    def push(self, state, action, reward, next_state, done, td_error=1.0):
        self.buffer.append((state, action, reward, next_state, done))
        # Higher TD error = higher priority
        priority = (abs(td_error) + 1e-6) ** self.alpha
        self.priorities.append(priority)
    
    def sample(self, batch_size: int, n_rows: int, n_cols: int, max_deliveries: int):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        
        # Sample based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Encode states
        states_enc = torch.stack([encode_state_normalized(s, n_rows, n_cols, max_deliveries) for s in states])
        next_states_enc = torch.stack([encode_state_normalized(s, n_rows, n_cols, max_deliveries) for s in next_states])
        
        return (
            states_enc,
            torch.tensor(actions, dtype=torch.int64, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            next_states_enc,
            torch.tensor(dones, dtype=torch.float32, device=device)
        )
    
    def __len__(self):
        return len(self.buffer)

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

class TrainingConfig:
    def __init__(self):
        # Network parameters
        self.state_dim = 3
        self.n_actions = 4
        self.hidden_size = 128
        
        # Training parameters
        self.buffer_capacity = 10000
        self.batch_size = 64  # Smaller batch size for faster training
        self.gamma = 0.99
        self.lr = 1e-3
        
        # Exploration parameters
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995  # Faster decay for quicker convergence
        
        # Training schedule
        self.target_update_freq = 100
        self.num_episodes = 1000  # Reduced for faster experimentation
        self.max_steps_per_episode = 500  # Reduced for medium grid
        
        # Logging
        self.log_interval = 25
        self.save_interval = 200

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_dqn(grid_name="medium_grid_new_3.npy", config=None):
    """Train DQN agent on medium grid with enhanced rewards."""
    
    if config is None:
        config = TrainingConfig()
    
    # Setup environment
    env = setup_environment(grid_name)
    if env is None:
        return None
    
    # Get environment info
    env.reset()
    n_rows, n_cols = env.grid.shape
    max_deliveries = env.initial_target_count
    
    print(f"Training on {grid_name}")
    print(f"Grid size: {n_rows}x{n_cols}")
    print(f"Max deliveries: {max_deliveries}")
    print(f"Using enhanced reward function")
    
    # Initialize networks
    policy_net = DQN(config.state_dim, config.n_actions, config.hidden_size).to(device)
    target_net = DQN(config.state_dim, config.n_actions, config.hidden_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # Initialize optimizer and replay buffer
    optimizer = optim.Adam(policy_net.parameters(), lr=config.lr)
    replay_buffer = PrioritizedReplayBuffer(config.buffer_capacity)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    success_rate = []
    recent_successes = deque(maxlen=50)  # Track recent successes
    
    epsilon = config.epsilon_start
    
    print("Starting training...")
    
    for episode in trange(1, config.num_episodes + 1, desc="Training"):
        raw_state = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(config.max_steps_per_episode):
            steps += 1
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randrange(config.n_actions)
            else:
                with torch.no_grad():
                    state_tensor = encode_state_normalized(raw_state, n_rows, n_cols, max_deliveries)
                    q_values = policy_net(state_tensor.unsqueeze(0))
                    action = q_values.argmax(dim=1).item()
            
            # Take action
            raw_next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Store transition
            replay_buffer.push(raw_state, action, reward, raw_next_state, done)
            
            raw_state = raw_next_state
            
            # Training step
            if len(replay_buffer) >= config.batch_size:
                # Sample batch
                states, actions, rewards, next_states, dones = replay_buffer.sample(
                    config.batch_size, n_rows, n_cols, max_deliveries
                )
                
                # Compute Q-values
                current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # Compute target Q-values
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(dim=1)[0]
                    target_q_values = rewards + config.gamma * next_q_values * (1 - dones)
                
                # Compute loss
                loss = F.mse_loss(current_q_values, target_q_values)
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
            
            if done:
                break
        
        # Track success (completed all deliveries and returned home)
        success = done and env.world_stats.get("final_bonus_given", 0) > 0
        recent_successes.append(success)
        
        # Decay epsilon
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        
        # Update target network
        if episode % config.target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Store metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        success_rate.append(np.mean(recent_successes) if recent_successes else 0)
        
        # Logging
        if episode % config.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-config.log_interval:])
            avg_length = np.mean(episode_lengths[-config.log_interval:])
            current_success_rate = success_rate[-1]
            
            print(f"Episode {episode:4d} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Avg Steps: {avg_length:5.1f} | "
                  f"Success Rate: {current_success_rate:.2f} | "
                  f"Epsilon: {epsilon:.3f}")
        
        # Save model periodically
        if episode % config.save_interval == 0:
            model_path = Path("models")
            model_path.mkdir(exist_ok=True)
            model_name = grid_name.replace('.npy', '_policy.pt')
            torch.save(policy_net.state_dict(), model_path / model_name)
    
    # Final save
    model_path = Path("models")
    model_path.mkdir(exist_ok=True)
    model_name = grid_name.replace('.npy', '_policy.pt')
    torch.save(policy_net.state_dict(), model_path / model_name)
    
    print(f"Training completed! Model saved as {model_name}")
    
    return {
        'policy_net': policy_net,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_rate': success_rate,
        'config': config
    }

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_agent(grid_name="medium_grid_new_3.npy", model_path=None, num_episodes=10):
    """Evaluate trained agent."""
    
    env = setup_environment(grid_name)
    if env is None:
        return None
    
    n_rows, n_cols = env.grid.shape
    max_deliveries = env.initial_target_count
    
    # Load model
    if model_path is None:
        model_path = Path("models") / grid_name.replace('.npy', '_policy.pt')
    
    policy_net = DQN(3, 4, 128).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()
    
    print(f"Evaluating agent on {grid_name} for {num_episodes} episodes...")
    
    results = []
    
    for episode in range(num_episodes):
        raw_state = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(500):
            steps += 1
            
            with torch.no_grad():
                state_tensor = encode_state_normalized(raw_state, n_rows, n_cols, max_deliveries)
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
    
    print(f"\nEvaluation Results:")
    print(f"Success Rate: {successes}/{num_episodes} ({successes/num_episodes*100:.1f}%)")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Average Targets Collected: {avg_targets:.1f}/{max_deliveries}")
    
    return results

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_results(training_results):
    """Plot training metrics."""
    
    episode_rewards = training_results['episode_rewards']
    episode_lengths = training_results['episode_lengths']
    success_rate = training_results['success_rate']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Results', fontsize=16)
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.6)
    axes[0, 0].plot(np.convolve(episode_rewards, np.ones(50)/50, mode='valid'), 'r-', linewidth=2)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    
    # Episode lengths
    axes[0, 1].plot(episode_lengths, alpha=0.6)
    axes[0, 1].plot(np.convolve(episode_lengths, np.ones(50)/50, mode='valid'), 'r-', linewidth=2)
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True)
    
    # Success rate
    axes[1, 0].plot(success_rate, 'g-', linewidth=2)
    axes[1, 0].set_title('Success Rate (50-episode window)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Success Rate')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True)
    
    # Reward distribution (last 200 episodes)
    last_rewards = episode_rewards[-200:] if len(episode_rewards) >= 200 else episode_rewards
    axes[1, 1].hist(last_rewards, bins=30, alpha=0.7)
    axes[1, 1].set_title('Reward Distribution (Last 200 episodes)')
    axes[1, 1].set_xlabel('Total Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    
    # Train the agent
    print("\n" + "="*60)
    print("TRAINING DQN AGENT ON MEDIUM GRID")
    print("="*60)
    
    config = TrainingConfig()
    training_results = train_dqn("medium_grid_new_3.npy", config)
    
    if training_results:
        # Plot results
        plot_training_results(training_results)
        
        # Evaluate the trained agent
        print("\n" + "="*60)
        print("EVALUATING TRAINED AGENT")
        print("="*60)
        
        evaluation_results = evaluate_agent("medium_grid_new_3.npy")
        
        print("\nTraining and evaluation completed!")
        print("You can now experiment with different grids and hyperparameters.")