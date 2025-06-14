from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
from itertools import product
import numpy as np

try:
    from world import Environment
    from agents.ppo_agent import PPOAgent
except ModuleNotFoundError:
    from os import path, pardir
    import sys

    root_path = path.abspath(path.join(
        path.abspath(__file__), pardir, pardir
    ))
    if root_path not in sys.path:
        sys.path.append(root_path)

    from world import Environment
    from agents.ppo_agent import PPOAgent


def parse_args():
    p = ArgumentParser(description="DIC PPO Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use.")
    p.add_argument("--no_gui", action="store_true")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--episodes", type=int, default=10_000)
    p.add_argument("--iter", type=int, default=500)
    p.add_argument("--random_seed", type=int, default=0)
    return p.parse_args()

def custom_reward_function(grid, new_pos, prev_pos, targets, start_pos):
    reward = -1  # default move cost

    cell_value = grid[new_pos]
    if cell_value in (1, 2):
        reward = -5  # obstacle
    elif cell_value == 3:
        reward = 10  # delivery spot

    if prev_pos is not None and cell_value != 3 and new_pos == prev_pos:
        reward -= 2  # penalize reversing except after target

        # Determine closest target or start
        target = None
        if targets:
            target = min(targets, key=lambda t: abs(new_pos[0] - t[0]) + abs(new_pos[1] - t[1]))
        else:
            target = start_pos

        # Compare distance from previous position and new position to target
        prev_dist = abs(prev_pos[0] - target[0]) + abs(prev_pos[1] - target[1])
        new_dist = abs(new_pos[0] - target[0]) + abs(new_pos[1] - target[1])

        if new_dist < prev_dist:
            reward += 2
        elif new_dist > prev_dist:
            reward -= 2

    return reward


def main(grid_paths, no_gui, episodes, max_steps, fps, sigma, random_seed):
    for grid in grid_paths:

        prev_pos = [None]  
        start_pos = [None]

        #learning_rates = [1e-4, 3e-4, 1e-3]
        #clip_epsilons = [0.1, 0.2, 0.3]
        #batch_sizes = [16, 32]
        #k_epochs_list = [3, 4, 5]
        #gammas = [0.95, 0.99]
        #gae_lambdas = [0.90, 0.95, 0.98]

        learning_rates = [3e-4]
        clip_epsilons = [0.2]
        batch_sizes = [32]
        k_epochs_list = [5]
        gammas = [0.99]
        gae_lambdas = [0.95]

        hyperparameter_combos = list(product(learning_rates, clip_epsilons, batch_sizes, k_epochs_list, gammas, gae_lambdas))

        best_score = -float("inf")
        best_params = None

        def reward_wrapper(grid_array, new_pos):
            targets = list(zip(*np.where(grid_array == 3)))
            reward = custom_reward_function(grid_array, new_pos, prev_pos[0], targets, start_pos[0])
            prev_pos[0] = new_pos
            return reward

        for lr, clip_eps, batch_size, k_epochs, gamma, gae_lambda in hyperparameter_combos:
            print(f"\nTraining with: lr={lr}, clip_eps={clip_eps}, batch_size={batch_size}, k_epochs={k_epochs}")

            # Reset environment each time
            env = Environment(grid, no_gui, sigma=sigma,
                              target_fps=fps, random_seed=random_seed,
                              reward_fn=reward_wrapper)

            agent = PPOAgent(
                lr=lr,
                gamma=gamma,
                clip_eps=clip_eps,
                gae_lambda=gae_lambda,
                k_epochs=k_epochs,
                batch_size=batch_size,
            )

            for ep in trange(episodes, desc="Training episodes", leave=False):
                state = env.reset()
                agent.grid_height, agent.grid_width = env.grid.shape
                agent.max_targets = env.initial_target_count

                start_pos[0] = env.agent_pos  
                prev_pos[0] = env.agent_pos  
                for _ in range(max_steps):
                    action = agent.take_action(state)
                    next_state, reward, terminated, info = env.step(action)

                    actual_action = info.get("actual_action", action)

                    agent.update(
                        next_state=next_state,
                        reward=reward,
                        action_taken=actual_action,
                        done=terminated
                    )

                    state = next_state
                    if terminated:
                        break

            # Evaluate agent performance
            score = Environment.evaluate_agent(
                grid, agent, max_steps, sigma,
                random_seed=random_seed
            )
            print(f"Score: {score:.2f} for lr={lr}, clip_eps={clip_eps}, batch_size={batch_size}, k_epochs={k_epochs}, gamma={gamma}, gae_lambda={gae_lambda}")

            if score > best_score:
                best_score = score
                best_params = (lr, clip_eps, batch_size, k_epochs, gamma, gae_lambda)

        print("\n=== Hyperparameter Search Complete ===")
        print(f"Best Score: {best_score:.2f}")
        print(f"Best Hyperparameters: lr={best_params[0]}, clip_eps={best_params[1]}, batch_size={best_params[2]}, k_epochs={best_params[3]}, gamma={best_params[4]}, gae_lambda={best_params[5]}")



if __name__ == '__main__':
    args = parse_args()
    main(
        args.GRID,
        args.no_gui,
        args.episodes,
        args.iter,
        args.fps,
        args.sigma,
        args.random_seed
    )
