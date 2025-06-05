from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange

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


def main(grid_paths, no_gui, episodes, max_steps, fps, sigma, random_seed):
    for grid in grid_paths:
        env = Environment(grid, no_gui, sigma=sigma,
                          target_fps=fps, random_seed=random_seed)

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

        # Optional evaluation after training
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