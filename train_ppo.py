from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange

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


def main(grid_paths, no_gui, episodes, max_steps, fps, sigma, random_seed):
    for grid in grid_paths:
        env = Environment(grid, no_gui, sigma=sigma,
                          target_fps=fps, random_seed=random_seed)

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
