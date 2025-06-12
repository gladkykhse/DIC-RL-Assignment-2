# train_dqn.py

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import numpy as np
from typing import Optional, Tuple

try:
    from world.delivery_environment import Environment
    from agents.dqn_agent import DQNAgent
except ModuleNotFoundError:
    from os import path, pardir
    import sys

    root_path = path.abspath(path.join(
        path.abspath(__file__), pardir, pardir
    ))
    if root_path not in sys.path:
        sys.path.append(root_path)
    from world.delivery_environment import Environment
    from agents.dqn_agent import DQNAgent


def parse_args():
    p = ArgumentParser(description="DIC DQN Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if no_gui is not set.")
    p.add_argument("--episodes", type=int, default=500,
               help="Number of training episodes to run.")
    p.add_argument("--iter", type=int, default=1000,
                   help="Number of steps to go through in a single episode.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    return p.parse_args()

def custom_reward(
    grid: np.ndarray,
    prev_pos: Optional[Tuple[int,int]],
    new_pos: Tuple[int,int],
    initial_target_count: int
) -> float:
    """
    - base reward: -1 on empty; -5 on obstacle; +10 on target
    - penalty -2 if new_pos == prev_pos
    - (initial_target_count is available for any extra logic you like)
    """
    cell = grid[new_pos]
    if cell == 0:
        base = -1.0
    elif cell in (1, 2):
        base = -10.0
    elif cell == 3:
        base = +10.0
    else:
        raise ValueError(f"Unexpected grid value {cell} at {new_pos}")

    # if stepping back one tile, apply extra penalty
    back_penalty = -5.0 if (prev_pos is not None and new_pos == prev_pos) else 0.0

    # (optional) use initial_target_count however you need—e.g. scale something. For now we ignore it.
    return base + back_penalty

def main(grid_paths: list[Path], no_gui: bool, episodes: int, iters: int, fps: int, sigma: float, random_seed: int):
    """Main loop for DQN training."""
    for grid in grid_paths:
        env = Environment(
            grid_fp=grid,
            no_gui=no_gui,
            sigma=sigma,
            reward_fn=custom_reward,
            target_fps=fps,
            random_seed=random_seed
        )

        agent = DQNAgent(
            state_dim=3,         # [x_norm, y_norm, r_norm]
            n_actions=4,         # up/down/left/right
            hidden_dim=64,
            lr=1e-3,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_capacity=10_000,
            batch_size=64,
            target_update_freq=500
        )

        for ep in range(episodes):
            state = env.reset()  # returns np.array([x_norm, y_norm, r_norm], dtype=float32)
            for _ in trange(iters, desc="Training steps"):
                action = agent.take_action(state)
                next_state, reward, done, info = env.step(action)

                # Update DQN with (state, action, reward, done, next_state)
                agent.update(
                    prev_state=state,
                    action=action,
                    reward=reward,
                    done=done,
                    next_state=next_state
                )

                state = next_state
                if done:
                    break

        Environment.evaluate_agent(
            grid_fp=grid,
            agent=agent,
            max_steps=iters,
            sigma=sigma,
            agent_start_pos=None,
            random_seed=random_seed,
            show_images=False
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
