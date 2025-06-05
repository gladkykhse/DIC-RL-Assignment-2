from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
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

    if prev_pos is not None:
        if new_pos == prev_pos:
            reward -= 2  # penalize reversing

        # Distance shaping
        if targets:
            prev_dists = [abs(prev_pos[0] - t[0]) + abs(prev_pos[1] - t[1]) for t in targets]
            new_dists = [abs(new_pos[0] - t[0]) + abs(new_pos[1] - t[1]) for t in targets]
        else:
            # All targets collected → guide agent back to start
            prev_dists = [abs(prev_pos[0] - start_pos[0]) + abs(prev_pos[1] - start_pos[1])]
            new_dists = [abs(new_pos[0] - start_pos[0]) + abs(new_pos[1] - start_pos[1])]

        if min(new_dists) < min(prev_dists):
            reward += 2
        elif min(new_dists) > min(prev_dists):
            reward -= 2

    return reward



def main(grid_paths, no_gui, episodes, max_steps, fps, sigma, random_seed):
    for grid in grid_paths:

        prev_pos = [None]  
        start_pos = [None]

        def reward_wrapper(grid_array, new_pos):
            targets = list(zip(*np.where(grid_array == 3)))
            reward = custom_reward_function(grid_array, new_pos, prev_pos[0], targets, start_pos[0])
            prev_pos[0] = new_pos
            return reward

        # Create environment with reward wrapper
        env = Environment(grid, no_gui, sigma=sigma,
                          target_fps=fps, random_seed=random_seed,
                          reward_fn=reward_wrapper)

        agent = PPOAgent(
            lr=3e-4,
            gamma=0.99,
            clip_eps=0.2,
            gae_lambda=0.95,
            k_epochs=4,
            batch_size=32,
        )


        for ep in trange(episodes, desc="Training episodes"):
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
