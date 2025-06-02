from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import numpy as np

try:
    from world.delivery_environment import Environment
    from agents.ppo import PPOAgent
except ModuleNotFoundError:
    from os import path, pardir
    import sys

    root_path = path.abspath(path.join(
        path.abspath(__file__), pardir, pardir
    ))
    if root_path not in sys.path:
        sys.path.append(root_path)
    from world.delivery_environment import Environment
    from agents.ppo import PPOAgent


def parse_args():
    p = ArgumentParser(description="DIC Q-Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use.")
    p.add_argument("--no_gui", action="store_true")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--iter", type=int, default=500)
    p.add_argument("--random_seed", type=int, default=0)
    return p.parse_args()


def main(grid_paths, no_gui, episodes, max_steps, fps, sigma, random_seed):
    for grid in grid_paths:
        env = Environment(grid, no_gui, sigma=sigma,
                          target_fps=fps, random_seed=random_seed)

        agent = PPOAgent(
            state_dim=3,
            action_dim=4,
            hidden_dim=128,
            gamma=0.99,
            clip_epsilon=0.2,
            lr=3e-4,
            update_epochs=4,
            batch_size=64,
            entropy_coef=0.01,
            value_coef=0.5
        )

        for _ in trange(episodes, desc="Training episodes"):
            state = env.reset()
            prev_pos = state[:2]

            for _ in range(max_steps):
                action = agent.take_action(state)
                next_state, env_reward, terminated, info = env.step(action)

                grid_array = env.grid
                curr_pos = next_state[:2]

                targets = (grid_array == 3).nonzero()
                if targets[0].size > 0:
                    targets_list = list(zip(targets[0], targets[1]))
                    prev_dists = [sum(abs(np.array(prev_pos) - np.array(t))) for t in targets_list]
                    curr_dists = [sum(abs(np.array(curr_pos) - np.array(t))) for t in targets_list]
                    shaping_reward = min(prev_dists) - min(curr_dists)
                else:
                    shaping_reward = 0

                shaped_reward = shaping_reward - 1
                if env_reward == 10:
                    shaped_reward = 10
                elif env_reward == -5:
                    shaped_reward = -5

                actual_action = info.get("actual_action", action)
                agent.store_transition(state, actual_action, shaped_reward, terminated, next_state)

                prev_pos = curr_pos
                state = next_state

                if terminated:
                    break
            
            agent.update()

        Environment.evaluate_agent(
            grid, agent, max_steps, sigma,
            random_seed=random_seed
        )


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