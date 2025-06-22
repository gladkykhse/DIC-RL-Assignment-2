# 2AMC15 - Data Intelligence Challenge | Assignment 2 

This repository contains the code for Assignment 2 of the Data Intelligence Challenge course. Building on the foundations from Task 1, it introduces:
- an updated delivery-task environment 
- new agent implementations 
- comprehensive training and evaluation scripts 
- our custom DQN and PPO models 
- a set of custom grids for both algorithms

These additions enable seamless experimentation with Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) approaches developed by our team.

## 🔧 Setup

1. **Clone the repo**  
```bash
git clone https://github.com/gladkykhse/DIC-RL-Assignment-2.git
cd DIC-RL-Assignment-2
```
2. Create & activate your environment with Python >= 3.10 (we use 3.11.11)
```bash
# with conda
conda create -n dic2025 python=3.11
conda activate dic2025

# or with venv
python -m venv venv
source venv/bin/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt 
```

## 🚀 Usage

### 1. DQN (Deep Q Network)
Repository already contains models trained for grids:
- `grid_configs/A1_grid.npy`
- `grid_configs/medium_grid_2.npy`
- `grid_configs/medium_grid_3.npy`

Our team developed a Python script `train_and_run_dqn.py` that includes a full training-and-evaluation driver for
a Deep Q-Network delivery agent: it sets up the grid environment with a fixed start position, trains a DQN using
prioritized replay and target-network updates, saves models, supports CLI hyper-parameter tuning across multiple
grid configs, and can plot or evaluate results on demand. 

To reproduce the evaluation results exactly as in the report you can execute:
```bash
python train_and_run_dqn.py <supported grid> --mode evaluate
```
In the evaluation mode the `agents/dqn_agent.py` file is used which defines a feed-forward Deep Q-Network
and a lightweight DQNAgent that normalizes grid-world states, loads a saved model, and selects actions
greedily via the network’s arg-max output.

To train a new model for a different grid you can execute:
```bash
python train_and_run_dqn.py <any grid> --mode train ...
```
Adjust any hyper-parameters on the command line (e.g., `--lr`, `--hidden_size`, `--batch_size`, `--fps`) to suit your experiment
the script will save the resulting policy under models/ using the grid name.

### 2. PPO (Proximal Policy Optimization)
Repository already contains models trained for grids:
- `grid_configs/small_grid.npy`
- `grid_configs/small_grid_2.npy`
- `grid_configs/custom_medium_grid_1.npy`
- `grid_configs/custom_medium_grid_2.npy`

You can run script `run_ppo.py` to reproduce evaluation results exactly as in the report as follows:
```bash
python run_ppo.py <supported grid>
```

In order to train a new model you can use script `train_ppo.py`:
```bash
python train_ppo.py <any grid> ...
```
where you can play with arguments and parameters of the script as well as grids to produce a new model. This script uses the agent from
`agents/ppo.py` which implements a Proximal Policy Optimization (PPO) agent for grid-based
delivery tasks, featuring separate policy and value networks, action-masking for invalid moves,
and full training logic with clipped-ratio loss, GAE, entropy regularization, and adaptive hyper-parameters.
