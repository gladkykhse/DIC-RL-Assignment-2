from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import numpy as np

try:
    from world.delivery_environment_ppo import Environment
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
    p = ArgumentParser(description="DIC PPO Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use.")
    p.add_argument("--no_gui", action="store_true")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--iter", type=int, default=500)
    p.add_argument("--random_seed", type=int, default=0)
    return p.parse_args()


def make_reward_function():
    visited = set()
    prev_pos = [None]
    start_pos = [None]
    world_stats = [None]
    step_counter = [0]

    def reward_fn(grid, agent_pos):
        nonlocal visited, prev_pos, start_pos, world_stats, step_counter
        reward = 0.0

        curr_pos = agent_pos
        all_targets_collected = (
            world_stats[0]["total_targets_reached"] == world_stats[0]["initial_target_count"]
        )
        returning_home = all_targets_collected and curr_pos != start_pos[0]

        match grid[agent_pos]:
            case 0:
                reward += -0.5
            case 1 | 2:
                reward -= 5
            case 3:
                reward += 10
            case _:
                raise ValueError(f"Invalid grid value at {agent_pos}: {grid[agent_pos]}")

        if not all_targets_collected:
            if curr_pos == prev_pos[0]:
                reward -= 0.4
            if curr_pos not in visited:
                reward += 1.0
            else:
                reward -= 0.1
            dist_from_start = abs(curr_pos[0] - start_pos[0][0]) + abs(curr_pos[1] - start_pos[0][1])
            reward += 0.05 * dist_from_start
            reward -= 0.002 * step_counter[0]
        elif returning_home:
            dist_to_start = abs(curr_pos[0] - start_pos[0][0]) + abs(curr_pos[1] - start_pos[0][1])
            reward += 5.0 / (dist_to_start + 1)
            reward -= 0.01 * step_counter[0]

        if world_stats[0]["total_failed_moves"] == 0:
            reward += 1.0
        else:
            reward -= 0.1 * world_stats[0]["total_failed_moves"]

        visited.add(curr_pos)
        prev_pos[0] = curr_pos
        step_counter[0] += 1
        return reward

    def initialize(env):
        visited.clear()
        prev_pos[0] = env.agent_pos
        start_pos[0] = env.start_pos
        world_stats[0] = env.world_stats
        step_counter[0] = 0

    reward_fn.initialize = initialize
    return reward_fn


def main(grid_paths, no_gui, episodes, max_steps, fps, sigma, random_seed):
    for grid in grid_paths:
        reward_fn = make_reward_function()
        env = Environment(grid, no_gui, sigma=sigma,
                          target_fps=fps, random_seed=random_seed,
                          reward_fn=reward_fn)

        agent = PPOAgent(
            state_dim=5,
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
            reward_fn.initialize(env)

            for _ in range(max_steps):
                action = agent.take_action(state)
                next_state, reward, terminated, info = env.step(action)

                actual_action = info.get("actual_action", action)
                agent.store_transition(state, actual_action, reward, terminated, next_state)

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