import random
from typing import List, Set, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import Counter
import copy


def visualize_regions(board: np.ndarray):
    """Visualize the regions and queens"""
    n = board.shape[0]
    
    # Create color map with distinct colors
    num_colors = len(set(board.flatten()))
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, num_colors))
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot regions
    im = ax.imshow(board, cmap=plt.cm.get_cmap('tab20'))
        
    # Customize the plot
    ax.grid(True, color='black', linewidth=0.5)
    ax.set_xticks(np.arange(-0.5, n, 1))
    ax.set_yticks(np.arange(-0.5, n, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    plt.title(f"Queen Positions and Regions on {n}x{n} Board")
    plt.tight_layout()
    plt.show()

def get_region_cells(board: np.ndarray, region: int) -> List[Tuple[int, int]]:
    """Get all cell coordinates belonging to a specific region."""
    return list(zip(*np.where(board == region)))

def get_adjacent_cells(pos: Tuple[int, int], board_size: int) -> Set[Tuple[int, int]]:
    """Get all adjacent cells (including diagonally adjacent)."""
    row, col = pos
    adjacent = set()
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < board_size and 0 <= new_col < board_size:
                adjacent.add((new_row, new_col))
    return adjacent

def is_valid_placement(queens: Set[Tuple[int, int]], new_pos: Tuple[int, int], board_size: int) -> bool:
    """Check if a new queen placement is valid."""
    if not queens:
        return True
    
    row, col = new_pos
    for qrow, qcol in queens:
        if row == qrow or col == qcol:
            return False
            
    adjacent_cells = get_adjacent_cells(new_pos, board_size)
    if any(queen in adjacent_cells for queen in queens):
        return False
        
    return True

def find_up_to_two_solutions(board: np.ndarray) -> List[List[Tuple[int, int]]]:
    """Find up to two solutions, returning queens' positions in sorted order."""
    board_size = len(board)
    regions = sorted(list(set(board.flatten())))

    # -1 indicates an uncolored square, remove it from candidates
    if -1 in regions:
        regions = regions[1:]

    region_cells = {region: get_region_cells(board, region) for region in regions}
    solutions = []
    
    def backtrack(current_region_idx: int, placed_queens: Set[Tuple[int, int]]):
        if len(solutions) >= 2:
            return
            
        if current_region_idx == len(regions):
            # Convert set to sorted list before adding to solutions
            solutions.append(sorted(list(placed_queens)))
            return
            
        region = regions[current_region_idx]
        for pos in region_cells[region]:
            if is_valid_placement(placed_queens, pos, board_size):
                placed_queens.add(pos)
                backtrack(current_region_idx + 1, placed_queens)
                placed_queens.remove(pos)
    
    backtrack(0, set())
    return solutions

if __name__ == "__main__":

    board_12_x_12 = np.array([
        [0, 0, 0, 2, 4, 4, 4, 4, 4, 4, 4, 4],
        [0, 0, 2, 2, 3, 3, 3, 4, 4, 3, 5, 4],
        [0, 0, 2, 2, 3, 3, 4, 4, 4, 3, 5, 5],
        [1, 0, 2, 2, 2, 3, 3, 3, 3, 3, 5, 6],
        [1, 0, 1, 2, 2, 2, 3, 3, 5, 5, 5, 6],
        [1, 1, 1, 1, 1, 2, 3, 5, 5, 5, 6, 6],
        [3, 3, 3, 3, 3, 3, 3, 3, 10, 10, 10, 6],
        [3, 7, 7, 7, 7, 3, 3, 10, 10, 10, 10, 6],
        [3, 7, 3, 7, 3, 3, 3, 3, 10, 3, 10, 11],
        [3, 3, 3, 7, 3, 8, 8, 3, 3, 3, 3, 11],
        [3, 3, 3, 7, 3, 8, 8, 9, 9, 9, 9, 9],
        [3, 3, 3, 7, 8, 8, 9, 9, 9, 9, 9, 9]
    ])

    # Queens positioning only
    # board_6_by_6 = np.array([
    #     [-1, -1, 3, -1, -1, -1],
    #     [-1, -1, -1, -1, -1, 4],
    #     [-1, 0, -1, -1, -1, -1],
    #     [-1, -1, -1, 5, -1, -1],
    #     [2, -1, -1, -1, -1, -1],
    #     [-1, -1, -1, -1, 1, -1]
    # ])


    board_6_by_6 = np.array([
        [-1, -1, 3, -1, -1, -1],
        [-1, -1, -1, -1, -1, 4],
        [2, 0, -1, -1, -1, -1],
        [2, 0, -1, 5, -1, -1],
        [2, 0, -1, -1, -1, -1],
        [-1, -1, -1, -1, 1, -1]
    ])

    board_6_by_6_one_move_in = np.array([
       [-1, -1, -1,  3,  3, -1],
       [-1, -1, -1, -1, -1,  1],
       [ 4, -1, -1, -1, -1, -1],
       [-1, -1,  0, -1, -1, -1],
       [-1, -1, -1, -1,  2, -1],
       [-1,  5, -1, -1, -1, -1]])

    solutions = find_up_to_two_solutions(board_6_by_6_one_move_in)

    if len(solutions) == 0:
        print("No solutions found!")
    elif len(solutions) == 1:
        print("Unique solution found!")
        print("Queen positions:", [(int(r), int(c)) for r,c in solutions[0]])
    else:
        print("Multiple solutions exist!")
        print("First solution:", [(int(r), int(c)) for r,c in solutions[0]])
        print("Second solution:", [(int(r), int(c)) for r,c in solutions[1]])

    # visualize_regions(board)
