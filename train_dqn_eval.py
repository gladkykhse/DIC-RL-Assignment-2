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
    max_targets = raw_state[2]
    input_dim = grid_shape[0] * grid_shape[1] + 1
    action_dim = 4
    agent = DQNAgent(input_dim, action_dim)

    epsilon = epsilon_start
    target_update_freq = 2

    all_rewards = []
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    for episode in range(episodes):
        raw_state = env.reset()
        if len(raw_state) != 3:
            raise ValueError(f"Expected 3-tuple state, got: {raw_state}")
        initial_target_count = raw_state[2]
        state = normalize_state(raw_state[:2], raw_state[2], grid_shape, initial_target_count)
        total_reward = 0

        for _ in range(max_steps):
            action = agent.act(state, epsilon)
            next_raw_state, reward, done, _ = env.step(action)
            next_state = normalize_state(next_raw_state[:2], next_raw_state[2], grid_shape, initial_target_count)

            if next_raw_state[2] < raw_state[2]:
                reward += 2  # scaled further down target reached bonus

            agent.buffer.push(state, action, reward, next_state, done)
            if len(agent.buffer) > 100:
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
        if episode % 50 == 0 or episode == episodes - 1:
            print(f"Episode {episode} | Total reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

    rewards_file = "results/dqn_rewards.npy"
    np.save(rewards_file, np.array(all_rewards))

    plt.plot(all_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Progress")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/dqn_reward_curve.png")
    print(f"Training complete. Rewards saved to {rewards_file}")

    return agent

def evaluate_dqn(agent, grid_path, episodes=50, max_steps=500):
    env = Environment(Path(grid_path), no_gui=False)
    raw_state = env.reset()
    grid_shape = env.grid.shape
    max_targets = raw_state[2]
    eval_rewards = []
    completions = 0

    for episode in range(episodes):
        raw_state = env.reset()
        initial_target_count = raw_state[2]
        state = normalize_state(raw_state[:2], raw_state[2], grid_shape, initial_target_count)
        total_reward = 0

        for _ in range(max_steps):
            if hasattr(env, "render"):
                env.render()
            action = agent.act(state, epsilon=0.0)
            next_raw_state, reward, done, _ = env.step(action)
            next_state = normalize_state(next_raw_state[:2], next_raw_state[2], grid_shape, initial_target_count)

            state = next_state
            total_reward += reward

            if done:
                completions += 1
                break

        eval_rewards.append(total_reward)

    avg_reward = np.mean(eval_rewards)
    completion_rate = completions / episodes * 100

    print(f"Evaluation complete. Avg reward: {avg_reward:.2f}, Completion rate: {completion_rate:.1f}%")
    np.save("results/dqn_eval_rewards.npy", np.array(eval_rewards))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=str, required=True, help="Path to grid file")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the trained agent after training")

    args = parser.parse_args()

    trained_agent = train_dqn(
        grid_path=args.grid,
        episodes=args.episodes,
        max_steps=args.max_steps,
    )

    if args.evaluate:
        evaluate_dqn(trained_agent, grid_path=args.grid, max_steps=args.max_steps)
