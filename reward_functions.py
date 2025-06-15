"""
Enhanced reward function for delivery robot environment.
This addresses the sparse reward problem by providing dense guidance rewards
based on Manhattan distance to targets and home position.
"""

import numpy as np
from typing import Tuple, List


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_target_positions(grid: np.ndarray) -> List[Tuple[int, int]]:
    """Get all target positions from the grid."""
    target_locs = np.where(grid == 3)
    return [(target_locs[0][i], target_locs[1][i]) for i in range(len(target_locs[0]))]


def min_manhattan_distance_to_targets(pos: Tuple[int, int], grid: np.ndarray) -> int:
    """Calculate minimum Manhattan distance from position to any target."""
    target_positions = get_target_positions(grid)
    if not target_positions:
        return 0
    distances = [manhattan_distance(pos, target_pos) for target_pos in target_positions]
    return min(distances)


def calculate_base_reward(grid: np.ndarray, pos: Tuple[int, int]) -> float:
    """Calculate base reward based on cell type (original logic)."""
    cell_value = grid[pos]
    
    if cell_value == 0:  # Empty tile
        return -1.0
    elif cell_value in [1, 2]:  # Wall or obstacle
        return -5.0
    elif cell_value == 3:  # Target tile
        return 10.0
    else:
        raise ValueError(f"Invalid grid cell value: {cell_value} at position {pos}")


def calculate_enhanced_reward(grid: np.ndarray, 
                            old_pos: Tuple[int, int], 
                            new_pos: Tuple[int, int], 
                            targets_remaining: int, 
                            start_pos: Tuple[int, int],
                            max_targets: int) -> float:
    """
    Enhanced reward function that provides dense rewards for navigation guidance.
    
    Args:
        grid: Current state of the grid
        old_pos: Agent's previous position
        new_pos: Agent's current position after taking action
        targets_remaining: Number of targets still to collect
        start_pos: Starting/home position for the agent
        max_targets: Initial number of targets
    
    Returns:
        Total reward for the action
    """
    
    # Base reward from original function
    base_reward = calculate_base_reward(grid, new_pos)
    
    # Initialize guidance reward
    guidance_reward = 0.0
    
    if targets_remaining > 0:
        # Phase 1: Still collecting targets
        old_min_dist = min_manhattan_distance_to_targets(old_pos, grid)
        new_min_dist = min_manhattan_distance_to_targets(new_pos, grid)
        
        if new_min_dist < old_min_dist:
            # Moved closer to nearest target
            guidance_reward += 0.5
        elif new_min_dist > old_min_dist:
            # Moved away from nearest target
            guidance_reward -= 0.2
            
    else:
        # Phase 2: All targets collected, returning home
        old_home_dist = manhattan_distance(old_pos, start_pos)
        new_home_dist = manhattan_distance(new_pos, start_pos)
        
        if new_home_dist < old_home_dist:
            # Moved closer to home
            guidance_reward += 1.0
        elif new_home_dist > old_home_dist:
            # Moved away from home
            guidance_reward -= 0.5
    
    # Efficiency bonus: reward for making progress overall
    total_progress_bonus = _calculate_progress_bonus(grid, new_pos, targets_remaining, 
                                                   start_pos, max_targets)
    
    total_reward = base_reward + guidance_reward + total_progress_bonus
    
    return total_reward


def _calculate_progress_bonus(grid: np.ndarray, 
                            pos: Tuple[int, int], 
                            targets_remaining: int,
                            start_pos: Tuple[int, int], 
                            max_targets: int) -> float:
    """Calculate bonus based on overall progress toward completion."""
    
    if targets_remaining == 0:
        # All targets collected - bonus based on proximity to home
        home_distance = manhattan_distance(pos, start_pos)
        max_distance = grid.shape[0] + grid.shape[1]  # Maximum possible distance
        
        # Inverse distance bonus (closer to home = higher bonus)
        proximity_bonus = 0.5 * (1.0 - home_distance / max_distance)
        return proximity_bonus
    
    else:
        # Still collecting - small bonus for overall efficiency
        targets_collected = max_targets - targets_remaining
        completion_ratio = targets_collected / max_targets
        
        # Small efficiency bonus
        return 0.1 * completion_ratio


# Usage example and testing
def test_reward_function():
    """Test the enhanced reward function with sample scenarios."""
    
    # Create a simple test grid
    test_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 3, 0, 3, 0],
        [0, 0, 0, 0, 0]
    ])
    
    start_pos = (0, 0)
    max_targets = 2
    
    print("Testing Enhanced Reward Function")
    print("=" * 40)
    
    # Test 1: Moving toward target
    old_pos = (0, 0)
    new_pos = (1, 0)  # Moving down toward targets
    targets_remaining = 2
    
    reward = calculate_enhanced_reward(test_grid, old_pos, new_pos, 
                                    targets_remaining, start_pos, max_targets)
    print(f"Test 1 - Moving toward target: {reward}")
    
    # Test 2: Moving away from target
    old_pos = (2, 1)
    new_pos = (1, 1)  # Moving up away from targets
    targets_remaining = 2
    
    reward = calculate_enhanced_reward(test_grid, old_pos, new_pos, 
                                    targets_remaining, start_pos, max_targets)
    print(f"Test 2 - Moving away from target: {reward}")
    
    # Test 3: Returning home after collecting all targets
    old_pos = (4, 4)
    new_pos = (3, 4)  # Moving toward home
    targets_remaining = 0
    
    reward = calculate_enhanced_reward(test_grid, old_pos, new_pos, 
                                    targets_remaining, start_pos, max_targets)
    print(f"Test 3 - Returning home: {reward}")
    
    print("\nReward function ready for integration!")


if __name__ == "__main__":
    test_reward_function()