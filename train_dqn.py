import os
import torch
import matplotlib.pyplot as plt
from agents.dqn_agent import DQNAgent
from world.delivery_environment import Environment
from pathlib import Path
import numpy as np


def one_hot_position(agent_pos, grid_shape):
    grid_size = grid_shape[0] * grid_shape[1]
    one_hot = np.zeros(grid_size)
    index = agent_pos[0] * grid_shape[1] + agent_pos[1]
    one_hot[index] = 1
    return one_hot

def normalize_state(agent_pos, remaining_targets, grid_shape, max_targets):
    
    one_hot_pos = one_hot_position(agent_pos, grid_shape)
    normalized_target = [remaining_targets / max_targets]
    return np.concatenate((one_hot_pos, normalized_target))

def train_dqn(grid_path, episodes=300, max_steps=500, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.99):
    env = Environment(Path(grid_path), no_gui=True)
    raw_state = env.reset()
    grid_shape = env.grid.shape
    input_dim = grid_shape[0] * grid_shape[1] + 1  
    action_dim = 4
    agent = DQNAgent(input_dim, action_dim)

    epsilon = epsilon_start
    target_update_freq = 10

    all_rewards = []
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    for episode in range(episodes):
        raw_state = env.reset()
        if len(raw_state) != 3:
            raise ValueError(f"Expected 3-tuple state, got: {raw_state}")
        state = normalize_state(raw_state[:2], raw_state[2], grid_shape, raw_state[2])
        total_reward = 0

        for _ in range(max_steps):
            action = agent.act(state, epsilon)
            next_raw_state, reward, done, _ = env.step(action)
            next_state = normalize_state(next_raw_state[:2], next_raw_state[2], grid_shape, raw_state[2])

            agent.buffer.push(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward

            if done:
                break

        
        if not done:
            total_reward -= 100

        all_rewards.append(total_reward)

        if episode % target_update_freq == 0:
            agent.update_target()
            torch.save(agent.model.state_dict(), f"checkpoints/dqn_episode_{episode}.pt")

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        print(f"Episode {episode} | Total reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

 

    return agent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=str, required=True, help="Path to grid file")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max_steps", type=int, default=500)

    args = parser.parse_args()

    train_dqn(
        grid_path=args.grid,
        episodes=args.episodes,
        max_steps=args.max_steps,
    )
