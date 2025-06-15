import numpy as np
import random
from agents import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(
        self,
        grid_fp,
        gamma=0.9,
        epsilon=0.1,
        alpha=0.1,
        convergence_tol=1e-3,
        patience=5,
    ):
        """
        Initializes the QLearningAgent with the given parameters.

        Args:
            grid_fp (str): File path to the NumPy grid configuration.
            gamma (float): Discount factor for future rewards. Defaults to 0.9.
            epsilon (float): Exploration rate for ε-greedy policy. Defaults to 0.1.
            alpha (float): Learning rate for Q-value updates. Defaults to 0.1.
            convergence_tol (float): Tolerance for convergence check based on Q-values. Defaults to 1e-3.
            patience (int): Number of consecutive stable steps for convergence. Defaults to 5.
        """
        super().__init__()
        self.gamma = gamma
        # self.epsilon = epsilon
        # decaying epsilon
        self.epsilon = epsilon
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995
        self.alpha = alpha
        self.convergence_tol = convergence_tol
        self.patience = patience

        self.grid = np.load(grid_fp)
        self.n_rows, self.n_cols = self.grid.shape
        self.max_deliveries = int(np.sum(self.grid == 3))

        self.actions = [0, 1, 2, 3]
        self.Q = np.zeros(
            (self.n_rows, self.n_cols, self.n_rows, self.n_cols, self.max_deliveries + 1, len(self.actions))
        )

        self._prev_Q = np.copy(self.Q)
        self._prev_policy = np.argmax(self.Q, axis=-1)
        self._unchanged_q_count = 0
        self._unchanged_policy_count = 0
        self.has_converged = False

    def take_action(self, state: tuple[int, int, int, int, int]) -> int:
        """
        Selects an action using an ε-greedy policy based on current Q-values.

        Args:
            state (tuple): The current state position as (row, column).

        Returns:
            int: The chosen action (0: right, 1: left, 2: up, 3: down).
        """
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        init_i, init_j, i, j, remaining = state
        # grab the Q-vector for this (row, col, remaining) tuple
        q_values = self.Q[init_i, init_j, i, j, remaining]
        max_q = np.max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def update(
        self,
        prev_state: tuple[int, int, int, int, int],
        action: int,
        reward: float,
        done: bool = False,
        next_state: tuple[int, int, int, int, int] = (0, 0),
    ):
        """
        Performs the Q-learning update for the current step based on the transition.

        Args:
            prev_state (tuple): The previous state as (row, column).
            action (int): The action taken in the previous state.
            reward (float): The reward received after the action.
            done (bool): Whether the episode has ended.
            next_state (tuple): The resulting state after the action.
        """
        init_i, init_j, i, j, rem = prev_state
        n_init_i, n_init_j, ni, nj, nrem = next_state
        next_q_max = 0 if done else np.max(self.Q[n_init_i, n_init_j, ni, nj, nrem])
        td_target = reward + self.gamma * next_q_max
        td_error = td_target - self.Q[init_i, init_j, i, j, rem, action]
        self.Q[i, j, rem, action] += self.alpha * td_error

        self._check_convergence()
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _check_convergence(self):
        """
        Checks for convergence by monitoring stability in Q-values and policy.

        If both the Q-values and the derived policy remain unchanged for a number
        of consecutive steps defined by `patience`, the agent is marked as converged.
        """
        # Q-table stability
        q_diff = np.max(np.abs(self.Q - self._prev_Q))
        if q_diff < self.convergence_tol:
            self._unchanged_q_count += 1
        else:
            self._unchanged_q_count = 0

        # Policy stability
        current_policy = np.argmax(self.Q, axis=-1)
        if np.array_equal(current_policy, self._prev_policy):
            self._unchanged_policy_count += 1
        else:
            self._unchanged_policy_count = 0

        # Update trackers
        self._prev_Q[:] = self.Q
        self._prev_policy[:] = current_policy

        # Converged if both have been stable
        if (
            self._unchanged_q_count >= self.patience
            and self._unchanged_policy_count >= self.patience
        ):
            self.has_converged = True
            print(
                f"Q-Learning Agent converged (tol={self.convergence_tol}, patience={self.patience})"
            )
