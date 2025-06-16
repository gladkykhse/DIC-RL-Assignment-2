from argparse import ArgumentParser
import argparse
from pathlib import Path
from tqdm import trange
import numpy as np
import torch

try:
    from world.delivery_environment import Environment
    from agents.ppo import PPOAgent
    from enhanced_reward_function import enhanced_reward_function
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
    p = ArgumentParser(description="DIC PPO Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use.")
    p.add_argument("--no_gui", action="store_true")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--agent_start_pos", type=parse_tuple, default=None)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--iter", type=int, default=500)
    p.add_argument("--random_seed", type=int, default=0)
    p.add_argument("--save_dir", type=Path, default=Path("models"))
    return p.parse_args()

def parse_tuple(s):
    try:
        x, y = map(int, s.split(","))
        return (x, y)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid format for --agent_start_pos: '{s}'. Expected format is 'x,y'.")

def make_reward_function():
    prev_pos = [None]
    start_pos = [None]
    max_targets = [None]

    def reward(grid, new_pos):
        cell_val = grid[new_pos]
        if cell_val in (1, 2):
            new_pos_effective = prev_pos[0]
        else:
            new_pos_effective = new_pos

        targets_remaining = int(np.sum(grid == 3))
        r = enhanced_reward_function(
            grid = grid,
            old_pos = prev_pos[0],
            new_pos = new_pos_effective,
            targets_remaining = targets_remaining,
            start_pos = start_pos[0],
            max_targets = max_targets[0]
        )

        prev_pos[0] = new_pos_effective
        return r

    def initialize(env):
        prev_pos[0] = env.agent_pos
        start_pos[0] = env.start_pos
        max_targets[0] = env.initial_target_count

    reward.initialize = initialize
    return reward


def main(grid_paths, no_gui, episodes, max_steps, fps, sigma, agent_start_pos, random_seed, save_dir):
    for grid in grid_paths:
        reward_fn = make_reward_function()
        env = Environment(grid, no_gui, sigma=sigma,
                          target_fps=fps, random_seed=random_seed,
                          reward_fn=reward_fn, agent_start_pos=agent_start_pos)
        
        state = env.reset()
        n_rows, n_cols = env.grid.shape
        max_targets = state[-1] 

        agent = PPOAgent(
            n_rows=n_rows,
            n_cols=n_cols,
            grid=env.grid,
            max_targets=max_targets,
            state_dim=5,
            action_dim=4,
            hidden_dim=256,
            gamma=0.99,
            clip_epsilon=0.2,
            lr=3e-4,
            update_epochs=10,
            batch_size=128,
            entropy_coef=0.01,
            value_coef=0.5
        )

        for _ in trange(episodes, desc="Training episodes"):
            state = env.reset()
            reward_fn.initialize(env)

            for _ in range(max_steps):
                action = agent.take_action(state)
                next_state, reward, terminated, info = env.step(action)

                agent.store_transition(state, action, reward, terminated, next_state)

                state = next_state

                if terminated:
                    break

            agent.update()
        
        ckpt = {
            "policy": agent.policy_net.state_dict(),
            "value":  agent.value_net.state_dict(),
            "n_rows": n_rows,
            "n_cols": n_cols,
            "grid_fp": str(env.grid_fp),
            "max_targets": max_targets
        }
        save_dir.mkdir(exist_ok=True, parents=True)
        torch.save(ckpt, save_dir / f"PPO_{grid.stem}.pt")

        # Environment.evaluate_agent(
        #     grid, agent, max_steps, sigma,
        #     random_seed=random_seed
        # )


if __name__ == '__main__':
    args = parse_args()
    main(
        args.GRID,
        args.no_gui,
        args.episodes,
        args.iter,
        args.fps,
        args.sigma,
        args.agent_start_pos,
        args.random_seed,
        args.save_dir
    )