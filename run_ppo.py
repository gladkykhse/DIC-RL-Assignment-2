from argparse import ArgumentParser
from pathlib import Path
import torch

try:
    from world import Environment, Grid
    from agents.ppo import PPOAgent
except ModuleNotFoundError:
    from os import path, pardir
    import sys

    root_path = path.abspath(path.join(path.abspath(__file__), pardir, pardir))
    if root_path not in sys.path:
        sys.path.append(root_path)
    from world import Environment


START_POSITIONS = {
    "small_grid.npy": (1, 2),
    "small_grid_2.npy": (1, 2),
    "custom_medium_grid_1.npy": (4, 6),
    "custom_medium_grid_2.npy": (4, 6),
}


def parse_args():
    p = ArgumentParser(description="PPO Agent Evaluator.")
    p.add_argument("GRID", type=Path, nargs="+", help="Paths to the grid file to use.")
    p.add_argument("--no_gui", action="store_true")
    p.add_argument("--sigma", type=float, default=0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--iter", type=int, default=500)
    p.add_argument("--random_seed", type=int, default=0)
    p.add_argument("--agent_start_pos", type=str)
    return p.parse_args()

def load_agent(ckpt_path, random_seed) -> PPOAgent:
    torch.manual_seed(random_seed)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    grid = Grid.load_grid(Path(ckpt["grid_fp"])).cells
    agent = PPOAgent(
        n_rows=ckpt["n_rows"],
        n_cols=ckpt["n_cols"],
        grid=grid,
        max_targets=ckpt["max_targets"],
    )
    agent.policy_net.load_state_dict(ckpt["policy"])
    agent.value_net.load_state_dict(ckpt["value"])
    return agent


def main(grid_paths, no_gui, max_steps, fps, sigma, random_seed, agent_start_pos):
    for grid_fp in grid_paths:
        ckpt_path = Path(f"models/PPO_{grid_fp.stem}.pt")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        agent = load_agent(ckpt_path, random_seed)

        if agent_start_pos:
            start_pos = tuple(map(int, agent_start_pos.split(",")))
        else:
            start_pos = START_POSITIONS.get(grid_fp.name, None)


        Environment.evaluate_agent(
            grid_fp,
            agent=agent,
            max_steps=max_steps,
            sigma=sigma,
            show_images=not no_gui,
            agent_start_pos=start_pos,
            random_seed=random_seed,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed, args.agent_start_pos)