import torch
import torch.multiprocessing as mp
from pathlib import Path
from time import sleep
import random
import glob

from agents.a3c_agent import ActorCriticNet, A3CAgent
from world import Environment

def train_worker(worker_id, grid_paths, global_model, optimizer,
                 no_gui, sigma, fps, episodes, max_steps, seed):
    torch.manual_seed(seed + worker_id)
    random.seed(seed + worker_id)

    agent = A3CAgent(global_model, optimizer, gamma=0.99)

    for ep in range(episodes):
        # Pick a random grid for this episode
        grid_path = random.choice(grid_paths)
        env = Environment(grid_path, no_gui, sigma=sigma, target_fps=fps,
                          random_seed=seed + worker_id + ep)

        state = env.reset()

        for step in range(max_steps):
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)
            actual_action = info.get("actual_action", action)

            agent.update(next_state, reward, actual_action, done)
            state = next_state

            if done:
                break

    print(f"[Worker {worker_id}] finished training.")


def main(grid_paths, no_gui, sigma, fps, episodes, max_steps, seed, num_workers):
    print(f"Training across {len(grid_paths)} grid(s) with {num_workers} parallel agents...")

    # Shared global model and optimizer
    global_model = ActorCriticNet().share_memory()
    optimizer = torch.optim.Adam(global_model.parameters(), lr=1e-4)

    # Spawn workers
    processes = []
    for worker_id in range(num_workers):
        p = mp.Process(target=train_worker,
                       args=(worker_id, grid_paths, global_model, optimizer,
                             no_gui, sigma, fps, episodes, max_steps, seed))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Training complete. Evaluating on each grid...")

    # Evaluate the final model on each grid
    for grid_path in grid_paths:
        print(f"Evaluating on: {grid_path.name}")
        agent = A3CAgent(global_model, optimizer)
        Environment.evaluate_agent(grid_path, agent, max_steps, sigma, random_seed=seed)


if __name__ == '__main__':
    import argparse
    mp.set_start_method('spawn')  # For Windows & PyTorch

    parser = argparse.ArgumentParser(description="Train A3C Agent on Multiple Grids")
    parser.add_argument("GRID", type=Path, nargs="+", help="Path(s) to grid .npy file(s)")
    parser.add_argument("--no_gui", action="store_true")
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--iter", type=int, default=300)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel agents")

    
    args = parser.parse_args()
    # Expand wildcard if necessary
    expanded_grids = []
    for g in args.GRID:
        if '*' in str(g):
            expanded_grids.extend(Path(p) for p in glob.glob(str(g)))
        else:
            expanded_grids.append(g)

    main(expanded_grids, args.no_gui, args.sigma, args.fps,
        args.episodes, args.iter, args.random_seed, args.workers)