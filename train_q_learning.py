from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange

try:
    from world import Environment
    from agents.q_learning import QLearningAgent
    from enhanced_reward_function import EnhancedRewardWrapper
except ModuleNotFoundError:
    from os import path, pardir
    import sys

    root_path = path.abspath(path.join(path.abspath(__file__), pardir, pardir))
    if root_path not in sys.path:
        sys.path.append(root_path)
    from world import Environment
    from agents.q_learning import QLearningAgent
    from enhanced_reward_function import EnhancedRewardWrapper


def parse_args():
    p = ArgumentParser(description="DIC Q-Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+", help="Paths to the grid file to use.")
    p.add_argument("--no_gui", action="store_true")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--episodes", type=int, default=10_000)
    p.add_argument("--iter", type=int, default=500)
    p.add_argument("--random_seed", type=int, default=0)
    p.add_argument("--enhanced_rewards", action="store_true")
    return p.parse_args()


def main(grid_paths, no_gui, episodes, max_steps, fps, sigma, random_seed, enhanced_rewards=False):
    for grid in grid_paths:
        # Create base environment
        base_env = Environment(grid, no_gui, sigma=sigma, target_fps=fps, random_seed=random_seed)
        
        # Optionally wrap with enhanced reward function
        if enhanced_rewards:
            env = EnhancedRewardWrapper(base_env)
        else:
            env = base_env

        agent = QLearningAgent(grid_fp=grid, gamma=0.9, epsilon=0.1, alpha=0.05, convergence_tol=1e-6, patience=50)

        for _ in trange(episodes, desc="Training episodes"):
            if agent.has_converged:
                break

            state = env.reset()
            for _ in range(max_steps):
                action = agent.take_action(state)
                next_state, reward, terminated, info = env.step(action)

                actual_action = info.get("actual_action", action)
                agent.update(
                    prev_state=state, action=actual_action, reward=reward, done=terminated, next_state=next_state
                )

                state = next_state
                if terminated or agent.has_converged:
                    break

        Environment.evaluate_agent(grid, agent, max_steps, sigma, random_seed=random_seed)


if __name__ == "__main__":
    args = parse_args()
    main(args.GRID, args.no_gui, args.episodes, args.iter, args.fps, args.sigma, args.random_seed, args.enhanced_rewards)
