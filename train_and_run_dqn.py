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
import argparse

# Add the project root to path if needed
if os.path.abspath('.') not in sys.path:
    sys.path.append(os.path.abspath('.'))

from world.delivery_environment import Environment
from reward_functions import calculate_base_reward, calculate_enhanced_reward
from agents.dqn_agent import DQN, DQNAgent

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

def setup_environment(grid_name, no_gui, target_fps):
    grid_path = Path(f"grid_configs/{grid_name}")
    
    if not grid_path.exists():
        print(f"Grid {grid_path} not found. Please run the grid generator first.")
        return None
    
    env = Environment(
        grid_path, 
        no_gui=no_gui, 
        target_fps=target_fps,
        reward_fn=calculate_base_reward,
        agent_start_pos=(4, 5)
    )
    
    return env

# ============================================================================
# STATE ENCODING
# ============================================================================

def encode_state_normalized(raw_state: tuple, n_rows: int, n_cols: int, max_deliveries: int) -> torch.Tensor:
    """
    Encode state with normalization for better training stability.
    """
    start_x, start_y, agent_x, agent_y, remaining = raw_state
    return torch.tensor([
        start_x / (n_rows - 1),
        start_y / (n_cols - 1),
        agent_x / (n_rows - 1),
        agent_y / (n_cols - 1),
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
        self.state_dim = 5
        self.n_actions = 4
        self.hidden_size = 128
        
        # Training parameters
        self.buffer_capacity = 5000  # Reduced from 10000
        self.batch_size = 32  # Reduced from 64 for faster training
        self.gamma = 0.99
        self.lr = 2e-3  # Slightly higher learning rate for faster convergence
        
        # Exploration parameters
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05  # Higher end value for more exploration
        self.epsilon_decay = 0.99  # Faster decay for quicker convergence
        
        # Training schedule
        self.target_update_freq = 50  # More frequent updates
        self.num_episodes = 300  # Significantly reduced from 1000
        self.max_steps_per_episode = 200  # Reduced from 500 for medium grid
        
        # Logging
        self.log_interval = 10  # More frequent logging
        self.save_interval = 100  # More frequent saves

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_dqn(grid_name="medium_grid.npy", config=None, no_gui=True, target_fps=30):
    """Train DQN agent on medium grid with enhanced rewards."""
    
    if config is None:
        config = TrainingConfig()
    
    # Setup environment
    env = setup_environment(grid_name, no_gui=no_gui, target_fps=target_fps)
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
            # Take action and get current position
            old_pos = env.agent_pos
            raw_next_state, original_reward, done, info = env.step(action)
            new_pos = env.agent_pos
              # Calculate enhanced reward
            targets_remaining = raw_next_state[4]  # Last element is remaining targets
            enhanced_reward = calculate_enhanced_reward(
                grid=env.grid,
                old_pos=old_pos,
                new_pos=new_pos,
                targets_remaining=targets_remaining,
                start_pos=env.start_pos,
                max_targets=max_deliveries
            )
            
            # Keep the original terminal bonus
            if done and env.world_stats.get("final_bonus_given", 0):
                enhanced_reward += 100  # Keep the big completion bonus
            
            total_reward += enhanced_reward
              # Store transition
            replay_buffer.push(raw_state, action, enhanced_reward, raw_next_state, done)
            
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
        'config': config    }

# ============================================================================
# GRID CONFIGURATION MAPPING
# ============================================================================

# Configuration mapping for different grids
GRID_MAPPING = {
    "grid_configs/small_grid.npy": {
        "model_file": "models/small_grid_policy.pt",
        "n_rows": 8,
        "n_cols": 8,
        "max_deliveries": 1,
        "hidden_size": 128,
    },
    "grid_configs/small_grid_2.npy": {
        "model_file": "models/small_grid_2_policy.pt",
        "n_rows": 8,
        "n_cols": 8,
        "max_deliveries": 2,
        "hidden_size": 128,
    },
    "grid_configs/A1_grid.npy": {
        "model_file": "models/A1_grid_policy.pt",
        "n_rows": 15,
        "n_cols": 15,
        "max_deliveries": 1,
        "hidden_size": 128,
    },    "grid_configs/medium_grid.npy": {
        "model_file": "models/medium_grid_policy.pt",
        "n_rows": 10,
        "n_cols": 10,
        "max_deliveries": 3,
        "hidden_size": 128,
    }
}

def get_grid_config(grid_path):
    """Get configuration for a specific grid file."""
    # Convert Windows backslashes to forward slashes for consistency
    grid_key = str(grid_path).replace('\\', '/')
    
    if grid_key in GRID_MAPPING:
        return GRID_MAPPING[grid_key]
    else:
        # Default configuration if not found in mapping
        print(f"Warning: No configuration found for {grid_key}. Using default config.")
        return {
            "model_file": f"models/{Path(grid_path).stem}_policy.pt",
            "n_rows": 10,
            "n_cols": 10,
            "max_deliveries": 3,
            "hidden_size": 128,
        }

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

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    """Parse command line arguments for flexible debugging and experimentation."""
    parser = argparse.ArgumentParser(description="DQN Training and Evaluation with Enhanced Rewards")
    # Environment parameters
    parser.add_argument("grids", nargs="*", type=str, default=["medium_grid.npy"],
                       help="Grid files to use for training/evaluation (default: medium_grid.npy)")
    parser.add_argument("--no_gui", action="store_true", default=False,
                       help="Disable GUI for faster training")
    parser.add_argument("--fps", type=int, default=10,
                       help="Frames per second for GUI (default: 10)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--sigma", type=float, default=0.0,
                       help="Environment noise parameter (default: 0.0)")
    
    # Training parameters
    parser.add_argument("--episodes", type=int, default=500,
                       help="Number of training episodes (default: 500)")
    parser.add_argument("--max_steps", type=int, default=200,
                       help="Maximum steps per episode (default: 200)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training (default: 32)")
    parser.add_argument("--lr", type=float, default=2e-3,
                       help="Learning rate (default: 2e-3)")
    parser.add_argument("--hidden_size", type=int, default=None,
                       help="Hidden layer size (default: use grid config)")
    
    # Modes
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "both"], default="both",
                       help="Mode: train, evaluate, or both (default: both)")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to pre-trained model for evaluation")
    parser.add_argument("--eval_episodes", type=int, default=10,
                       help="Number of episodes for evaluation (default: 10)")
    parser.add_argument("--iter", type=int, default=100,
                       help="Maximum steps for evaluation (default: 100)")
    
    # Debug options
    parser.add_argument("--debug", action="store_true", default=False,
                       help="Enable debug mode with reduced episodes for testing")
    parser.add_argument("--plot", action="store_true", default=True,
                       help="Show training plots (default: True)")

    return parser.parse_args()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    
    # Adjust parameters for debug mode
    if args.debug:
        print("DEBUG MODE: Reducing episodes and enabling GUI for testing")
        args.episodes = 30  # Even faster for debug
        args.max_steps = 100  # Reduced steps for debug
        args.no_gui = False
        args.fps = 8  # Slightly faster FPS
    
    # Process each grid
    for grid_name in args.grids:
        print(f"\n{'='*80}")
        print(f"PROCESSING GRID: {grid_name}")
        print(f"{'='*80}")
        
        # Get grid configuration
        grid_path = Path(f"grid_configs/{grid_name}")
        if not grid_path.exists():
            print(f"Error: Grid file {grid_path} not found. Skipping...")
            continue
            
        grid_config = get_grid_config(grid_path)
        
        # Create custom config based on arguments and grid config
        config = TrainingConfig()
        config.num_episodes = args.episodes
        config.max_steps_per_episode = args.max_steps
        config.batch_size = args.batch_size
        config.lr = args.lr
        # Use provided hidden_size or grid config default
        config.hidden_size = args.hidden_size if args.hidden_size is not None else grid_config["hidden_size"]
        
        print(f"Configuration:")
        print(f"  Grid: {grid_name}")
        print(f"  Model file: {grid_config['model_file']}")
        print(f"  Grid size: {grid_config['n_rows']}x{grid_config['n_cols']}")
        print(f"  Max deliveries: {grid_config['max_deliveries']}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  GUI: {'Disabled' if args.no_gui else f'Enabled ({args.fps} FPS)'}")
        print(f"  Episodes: {config.num_episodes}")
        print(f"  Max steps per episode: {config.max_steps_per_episode}")
        print(f"  Mode: {args.mode}")
        
        # Training
        if args.mode in ["train", "both"]:
            print("\n" + "="*60)
            print("TRAINING DQN AGENT")
            print("="*60)
            
            training_results = train_dqn(
                grid_name=grid_name,
                config=config,
                no_gui=args.no_gui,
                target_fps=args.fps
            )
            
            if training_results and args.plot:
                # Plot results
                plot_training_results(training_results)
          # Evaluation
        if args.mode in ["evaluate", "both"]:
            print("\n" + "="*60)
            print("EVALUATING TRAINED AGENT")
            print("="*60)
            
            # Use provided model path or grid config default
            model_path = args.model_path
            if model_path is None:
                model_path = Path(grid_config["model_file"])
            
            # Create DQN agent for evaluation
            agent = DQNAgent(
                model_file=str(model_path),
                hidden_size=grid_config["hidden_size"],
                device=device,
                n_rows=grid_config["n_rows"],
                n_cols=grid_config["n_cols"],
                max_deliveries=grid_config["max_deliveries"]
            )

            Environment.evaluate_agent(
                grid_fp=Path(f"grid_configs/{grid_name}"),
                agent=agent,
                max_steps=args.max_steps,
                sigma=args.sigma,
                agent_start_pos=(4, 5),
                random_seed=args.random_seed,
                show_images=False
            )
    
    print(f"\n{'='*80}")
    print(f"EXECUTION COMPLETED! Mode: {args.mode}")
    print("You can now experiment with different grids and hyperparameters.")
    print(f"{'='*80}")