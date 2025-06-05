from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange

try:
    from world import Environment
    from agents.dqn_agent import DQNAgent
except ModuleNotFoundError:
    from os import path, pardir
    import sys

    root_path = path.abspath(path.join(path.abspath(__file__), pardir, pardir))
    if root_path not in sys.path:
        sys.path.append(root_path)
    from world import Environment
    from agents.q_learning import QLearningAgent


def parse_args():
    p = ArgumentParser(description="DIC Q-Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+", help="Paths to the grid file to use.")
    p.add_argument("--no_gui", action="store_true")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--iter", type=int, default=500)
    p.add_argument("--random_seed", type=int, default=0)
    return p.parse_args()


MAPPING = {
    "grid_configs/small_grid.npy": {
        "model_file": "models/small_grid_policy.pt",
        "hidden_size": 64,
        "n_rows": 8,
        "n_cols": 8,
        "max_deliveries": 1,
    },
    "grid_configs/small_grid_2.npy": {
        "model": "models/small_grid_2_policy.pt",
        "hidden_size": 64,
        "model_file": 8,
        "n_cols": 8,
        "max_deliveries": 2,
    },
}


def main(grid_paths, no_gui, max_steps, fps, sigma, random_seed):
    for grid in grid_paths:
        env = Environment(grid, no_gui, sigma=sigma, target_fps=fps, random_seed=random_seed)
        env.reset()

        agent = DQNAgent(
            model_file=MAPPING[grid]["model_file"],
            hidden_size=MAPPING[grid]["hidden_size"],
            device="cpu",
            n_rows=MAPPING[grid]["n_rows"],
            n_cols=MAPPING[grid]["n_cols"],
            max_deliveries=MAPPING[grid]["max_deliveries"],
        )

        Environment.evaluate_agent(grid, agent, max_steps, sigma, random_seed=random_seed)


if __name__ == "__main__":
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed)
