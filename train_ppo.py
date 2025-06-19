from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import numpy as np

try:
    from world.delivery_environment import Environment
    from agents.ppo_agent import PPOAgent
except ModuleNotFoundError:
    from os import path, pardir
    import sys

    root_path = path.abspath(path.join(
        path.abspath(__file__), pardir, pardir
    ))
    if root_path not in sys.path:
        sys.path.append(root_path)
    from world.delivery_environment import Environment
    from agents.ppo_agent import PPOAgent

def parse_args():
    p = ArgumentParser(description="PPO Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use.")
    p.add_argument("--no_gui", action="store_true")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--iter", type=int, default=500)
    p.add_argument("--random_seed", type=int, default=0)
    return p.parse_args()

def build_reward_function():
    visited_cells = set()
    last_pos = [None]
    home_pos = [None]
    stats_ref = [None]
    step_num = [0]

    def reward(grid, agent_pos):
        nonlocal visited_cells, last_pos, home_pos, stats_ref, step_num
        r = 0.0
        cell_value = grid[agent_pos]
        stats = stats_ref[0]

        all_delivered = (stats["total_targets_reached"] == stats["initial_target_count"])
        returning_to_home = all_delivered and (agent_pos != home_pos[0])

        # Base penalties and rewards by cell type
        if cell_value == 0:
            r -= 0.4  # neutral cell - small penalty to encourage progress
        elif cell_value in (1, 2):
            r -= 4.0  # obstacle penalty
        elif cell_value == 3:
            r += 12.0  # delivery reward
        else:
            raise ValueError(f"Unknown grid value {cell_value} at {agent_pos}")

        # Encourage exploration before deliveries complete
        if not all_delivered:
            if agent_pos == last_pos[0]:
                r -= 0.3  # discourage idling
            if agent_pos not in visited_cells:
                r += 0.9  # encourage new cells
            else:
                r -= 0.15  # discourage revisits

            # Reward a bit more the further away from home to promote coverage
            manhattan_dist = abs(agent_pos[0] - home_pos[0][0]) + abs(agent_pos[1] - home_pos[0][1])
            r += 0.04 * manhattan_dist

            # Slightly penalize longer episodes
            r -= 0.0015 * step_num[0]

        # After all deliveries, encourage returning home quickly
        elif returning_to_home:
            manhattan_dist = abs(agent_pos[0] - home_pos[0][0]) + abs(agent_pos[1] - home_pos[0][1])
            r += 4.0 / (manhattan_dist + 1)
            r -= 0.007 * step_num[0]

        # Penalize failed moves proportionally
        failed_moves = stats.get("total_failed_moves", 0)
        if failed_moves > 0:
            r -= 0.12 * failed_moves
        else:
            r += 0.8  # bonus for flawless moves

        # Update state trackers
        visited_cells.add(agent_pos)
        last_pos[0] = agent_pos
        step_num[0] += 1

        return r

    def initialize(env):
        visited_cells.clear()
        last_pos[0] = env.agent_pos
        home_pos[0] = env.start_pos
        stats_ref[0] = env.world_stats
        step_num[0] = 0

    reward.initialize = initialize
    return reward


def main(grid_paths, no_gui, episodes, max_steps, fps, sigma, random_seed):
    for grid in grid_paths:
        reward = build_reward_function()
        env = Environment(grid, no_gui, sigma=sigma,
                          target_fps=fps, random_seed=random_seed, reward_fn=reward)

        agent = PPOAgent(
            state_dim=3,
            action_dim=4,
            lr=3e-4,
            gamma=0.99,
            k_epochs=4,
            gae_lambda=0.95, 
            batch_size=32, 
            entropy_coef=0.01, 
            value_coef=0.5, 
            max_grad_norm=0.5, 
            clip_eps=0.2
        )

        for _ in trange(episodes, desc="Training episodes"):
            state = env.reset()
            reward.initialize(env)
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

        # Optional --> evaluation after training
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


'''Changes TODO:
- Instead of one-hot encoding to represent grid cell indices, if you represent grid positions or indices directly as real-valued coordinates (e.g., using (x, y) as floats or integers), 
  then the state representation becomes continuous. Neural networks can generalize better with such inputs, because small changes in position lead to small changes in input values. 
- Include the number of remaining deliveries since it is a critical part of the environment's state. Without this, the agent can't distinguish between different phases of the task 
  (e.g., “should I still deliver?” vs. “should I return?”).  
- Implement a reward function (for instance, for the distances)
- Normalizing indices'''