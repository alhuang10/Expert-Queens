import random
from typing import List, Set, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import Counter
from get_solutions import find_up_to_two_solutions
import argparse
from collections import defaultdict
from itertools import combinations


def visualize_queens(positions: List[Tuple[int, int]], n: int = 8):
    """Visualize queen positions on a chess board using matplotlib"""
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create chess board pattern
    board = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 0:
                board[i, j] = 0.8  # Light squares
            else:
                board[i, j] = 0.3  # Dark squares
    
    # Plot the board
    ax.imshow(board, cmap='gray')
    
    # Plot queens as red dots with white edge
    queen_rows, queen_cols = zip(*positions)
    ax.scatter(queen_cols, queen_rows, color='red', s=300, marker='o', 
              edgecolor='white', linewidth=2, zorder=2, label='Queens')
    
    # Customize the plot
    ax.grid(True, color='black', linewidth=0.5)
    ax.set_xticks(np.arange(-0.5, n, 1))
    ax.set_yticks(np.arange(-0.5, n, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Add board coordinates
    for i in range(n):
        ax.text(-0.7, i, str(i), ha='center', va='center')
        ax.text(i, -0.7, str(i), ha='center', va='center')
    
    plt.title(f"Queen Positions on {n}x{n} Board")
    plt.tight_layout()
    plt.show()



def generate_random_queens(n: int = 8) -> List[Tuple[int, int]]:

    start_time = time.time()

    """Generate random valid queen positions on an nxn board using backtracking"""
    def is_valid_position(pos: Tuple[int, int], queens: List[Tuple[int, int]]) -> bool:
        row, col = pos
        
        # Check if position shares row or column with existing queens
        for q_row, q_col in queens:
            if row == q_row or col == q_col:
                return False
            
        # Check if position is adjacent to existing queens (including diagonally)
        for q_row, q_col in queens:
            if abs(row - q_row) <= 1 and abs(col - q_col) <= 1:
                return False
                
        return True

    def backtrack(queens: List[Tuple[int, int]], 
                 remaining_positions: List[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
        if len(queens) == n:
            return queens
            
        # Shuffle remaining positions for randomness
        positions = remaining_positions.copy()
        random.shuffle(positions)
        
        for pos in positions:
            if is_valid_position(pos, queens):
                queens.append(pos)
                new_remaining = [p for p in positions if p != pos]
                result = backtrack(queens, new_remaining)
                if result is not None:
                    return result
                queens.pop()
                
        return None

    # Initialize all possible positions
    all_positions = [(i, j) for i in range(n) for j in range(n)]
    
    # Keep trying until we find a valid solution
    while True:
        result = backtrack([], all_positions)
        if result is not None:
            # print(f"Getting a queen arrangement took {time.time() - start_time} sec")
            return result

from typing import List, Tuple
import numpy as np
import random

def score_queen_configuration(queens: List[Tuple[int, int]], n: int) -> float:
    """
    Score a queen configuration based on properties that might lead to unique color solutions.
    Higher score is better.
    """
    score = 0.0
    
    # Helper to get distances between queens
    def manhattan_dist(q1: Tuple[int, int], q2: Tuple[int, int]) -> int:
        return abs(q1[0] - q2[0]) + abs(q1[1] - q2[1])
    
    def diagonal_dist(q1: Tuple[int, int], q2: Tuple[int, int]) -> int:
        return max(abs(q1[0] - q2[0]), abs(q1[1] - q2[1]))
    
    # 1. Score clustering - we want some clustering but not too much
    distances = []
    for i, q1 in enumerate(queens):
        for j, q2 in enumerate(queens):
            if i < j:
                dist = manhattan_dist(q1, q2)
                distances.append(dist)
    
    # Prefer a mix of distances - calculate standard deviation
    # Higher std dev means more variety in spacing
    if distances:
        distance_std = np.std(distances)
        score += distance_std * 0.5
    
    # 2. Score diagonal relationships
    # diagonal_pairs = 0
    # for i, q1 in enumerate(queens):
    #     for j, q2 in enumerate(queens):
    #         if i < j:
    #             if abs(q1[0] - q2[0]) == abs(q1[1] - q2[1]):
    #                 diagonal_pairs += 1
    # score += diagonal_pairs * 0.3  # Reward some diagonal relationships
    
    # 3. Score edge/center balance
    edge_queens = sum(1 for r, c in queens if r in [0, n-1] or c in [0, n-1])
    ideal_edge_queens = len(queens) / 3  # Want roughly 1/3 on edges
    score -= abs(edge_queens - ideal_edge_queens) * 0.4
    
    # 4. Penalize too-uniform spacing
    all_dists = set(distances)
    if len(all_dists) < len(queens) / 2:  # Too few unique distances
        score -= 1.0
        
    return score

def generate_queens_heuristic(n: int, num_attempts: int = 100) -> List[Tuple[int, int]]:
    """
    Generate a queen configuration optimized for unique color solutions.
    """
    def is_valid_position(queens: List[Tuple[int, int]], new_pos: Tuple[int, int]) -> bool:
        """Check if new position violates queen placement rules"""
        row, col = new_pos
        for qr, qc in queens:
            # Check row/column conflicts
            if row == qr or col == qc:
                return False
            # Check adjacency (including diagonal)
            if abs(row - qr) <= 1 and abs(col - qc) <= 1:
                return False
        return True
    
    best_queens = None
    best_score = float('-inf')
    
    for _ in range(num_attempts):
        queens = []
        available_positions = [(r, c) for r in range(n) for c in range(n)]
        random.shuffle(available_positions)
        
        # Place queens one at a time
        while len(queens) < n and available_positions:
            placed = False
            for pos in available_positions[:]:
                if is_valid_position(queens, pos):
                    queens.append(pos)
                    available_positions.remove(pos)
                    placed = True
                    break
            
            if not placed:
                break
        
        # Score this configuration if we placed all queens
        if len(queens) == n:
            score = score_queen_configuration(queens, n)
            if score > best_score:
                best_score = score
                best_queens = queens.copy()
    
    if best_queens is None:
        raise ValueError("Could not find valid queen configuration")
        
    return best_queens


# Verify the solution
def verify_solution(positions: List[Tuple[int, int]], n: int) -> bool:
    """Verify that the solution meets all constraints"""
    if len(positions) != n:
        return False
        
    rows = set()
    cols = set()
    
    for i, (row, col) in enumerate(positions):
        # Check row and column constraints
        if row in rows or col in cols:
            return False
        rows.add(row)
        cols.add(col)
        
        # Check adjacency constraints
        for j, (other_row, other_col) in enumerate(positions):
            if i != j:
                if abs(row - other_row) <= 1 and abs(col - other_col) <= 1:
                    return False
                    
    return True

def generate_regions(queens: List[Tuple[int, int]], n: int = 8) -> np.ndarray:
    """
    Generate contiguous colored regions on the board, one per queen.
    Returns an nxn numpy array where each cell contains a number 0 to n-1 representing 
    its region.
    """
    start_time = time.time()

    # Initialize board with -1 (uncolored)
    board = np.full((n, n), -1)
    
    # Assign random colors to queens
    colors = list(range(len(queens)))
    random.shuffle(colors)
    for (row, col), color in zip(queens, colors):
        board[row, col] = color
    
    def get_adjacent_cells(row: int, col: int) -> List[Tuple[int, int]]:
        """Get orthogonally adjacent cells (no diagonals)"""
        adjacent = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:  # only up, down, left, right
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < n and 0 <= new_col < n:
                adjacent.append((new_row, new_col))
        return adjacent
    
    def get_candidate_cells() -> Set[Tuple[int, int]]:
        """Get all uncolored cells that are adjacent to colored cells"""
        candidates = set()
        for i in range(n):
            for j in range(n):
                if board[i, j] == -1:  # if uncolored
                    # Check if adjacent to any colored cell
                    adjacent = get_adjacent_cells(i, j)
                    if any(board[r, c] != -1 for r, c in adjacent):
                        candidates.add((i, j))
        return candidates
    
    def get_adjacent_colors(row: int, col: int) -> Set[int]:
        """Get all colors adjacent to a cell"""
        colors = set()
        for adj_row, adj_col in get_adjacent_cells(row, col):
            color = board[adj_row, adj_col]
            if color != -1:
                colors.add(color)
        return colors
    
    # Main coloring loop
    while True:
        candidates = get_candidate_cells()
        if not candidates:
            break
            
        # Pick random candidate
        row, col = random.choice(list(candidates))
        
        # Get adjacent colors and pick one randomly
        adj_colors = get_adjacent_colors(row, col)
        if adj_colors:
            chosen_color = random.choice(list(adj_colors))
            board[row, col] = chosen_color
    
    # print(time.time() - start_time)

    return board

def generate_regions_jagged(queens: List[Tuple[int, int]], n: int = 8) -> np.ndarray:
    """
    Generate non-compact, jagged regions to increase likelihood of unique solutions.
    Returns an nxn numpy array where each cell contains a number 0 to n-1 representing its region.
    """
    # Initialize board with -1 (uncolored)
    board = np.full((n, n), -1)
    
    # Assign random colors to queens
    colors = list(range(len(queens)))
    color_to_queen_loc = {}

    random.shuffle(colors)
    for (row, col), color in zip(queens, colors):
        board[row, col] = color
        color_to_queen_loc[color] = (row, col)

    def get_adjacent_cells(row: int, col: int) -> List[Tuple[int, int]]:
        """Get orthogonally adjacent cells"""
        adjacent = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < n and 0 <= new_col < n:
                adjacent.append((new_row, new_col))
        return adjacent
    
    def get_adjacent_cells_diag(row: int, col: int) -> List[Tuple[int, int]]:
        """Get all adjacent cells including diagonals"""
        adjacent = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr != 0 or dc != 0:
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < n and 0 <= new_col < n:
                        adjacent.append((new_row, new_col))
        return adjacent
    
    def softmax(scores: List[float], temperature: float = 1.0) -> List[float]:
        """
        Convert scores to probabilities using softmax with temperature.
        Higher temperature = more random, Lower = more deterministic
        """
        # Shift scores to avoid overflow
        scores = np.array(scores)
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores / temperature)
        return exp_scores / exp_scores.sum()

    
    def get_spindly_score(row: int, col: int, color: int, queens,
                          square_to_disallowed_color_mapping) -> float:
        """
        Calculate how 'spindly' placing color at this position would be.
        Higher score = more jagged/spindly (preferred)
        """
        adjacent = get_adjacent_cells(row, col)
        
        # Count colored neighbors of same and different colors
        same_color = 0
        diff_color = 0
        for adj_row, adj_col in adjacent:
            if board[adj_row, adj_col] == color:
                same_color += 1
            elif board[adj_row, adj_col] != -1:
                diff_color += 1

        found_banned_square_adjacent = False
        neighbors_of_potential_spot = get_adjacent_cells(row, col)

        for n in neighbors_of_potential_spot:
            if color in square_to_disallowed_color_mapping[n]:
                found_banned_square_adjacent = True

        color_banned_penalty = 0
        if found_banned_square_adjacent:
            color_banned_penalty = -10
        

        # Prefer positions that:
        # 1. Have few neighbors of same color (creates thin regions)
        # 2. Have many neighbors of different colors (creates jagged boundaries)
        # 3. Not marking next to a banned color
        return diff_color - (same_color * 1.5) + color_banned_penalty
    
    # Keep track of uncolored cells
    uncolored_cells = set((i, j) for i in range(n) for j in range(n) 
                         if board[i, j] == -1)
    
    # Using symmetry test to mark disallowed colors
    square_to_disallowed_color_mapping = defaultdict(list)  # (row, col) -> list(int)

    while uncolored_cells:
        # Find all uncolored cells adjacent to colored regions
        candidates = []
        for row, col in uncolored_cells:
            adjacent = get_adjacent_cells(row, col)
            adj_colors = set(board[r, c] for r, c in adjacent if board[r, c] != -1)
            
            # For each adjacent color, calculate spindly score
            for color in adj_colors:
                score = get_spindly_score(row, col, color, queens, square_to_disallowed_color_mapping)
                candidates.append((score, (row, col, color)))
        
        if not candidates:
            # We should never reach here            
            import pdb; pdb.set_trace()
            # No cells adjacent to colored regions, pick random color adjacent to any uncolored cell
            pos = random.choice(list(uncolored_cells))
            adjacent = get_adjacent_cells(*pos)
            adj_colors = [board[r, c] for r, c in adjacent if board[r, c] != -1]
            if adj_colors:
                color = random.choice(adj_colors)
            else:
                color = random.choice(colors)  # Completely isolated cell
            board[pos[0], pos[1]] = color
            uncolored_cells.remove(pos)
        else:
            next_color_found = False        

            while not next_color_found:
                # Termination condition: if none of the candidates work then return 
                #   None and try again with new starting state in the outer loop
                if not candidates:
                    # visualize_regions_queens(board, queens)
                    print("Went through all candidates, restarting by returning None!")
                    return None
                
                # Method 1: Chose a candidate from ones with the highest score
                # max_score = max(score for score, _ in candidates)
                # best_candidates = [(score, (r, c, color)) for score, (r, c, color) in candidates 
                #                 if score == max_score]
                # selected_score, (row, col, color) = random.choice(best_candidates)
                # print(f"Selected Score: {selected_score}")

                # Method 2: Sample from candidates
                # scores = [score for score, _ in candidates]
                scores = [score for score, _ in candidates]
                positions = [(r, c, color) for _, (r, c, color) in candidates]
                probabilities = softmax([x+2 for x in scores], temperature=0.2)
    
                # Choose based on probabilities instead of just the max
                selected_idx = np.random.choice(len(positions), p=probabilities)
                row, col, color = positions[selected_idx]
                selected_score = scores[selected_idx]
                
                # See if the color at that position is banned square color 
                if color in square_to_disallowed_color_mapping[(row, col)]:
                    candidates.remove((selected_score, (row, col, color)))
                    # print("Rejected candidate due to color, next choice")
                    continue

                # Test if the resulting board is single solution
                board[row, col] = color

                solutions = find_up_to_two_solutions(board)

                if len(solutions) == 1:
                    # If so, mark next_color_found as True and visualize
                    next_color_found = True
                    uncolored_cells.remove((row, col))

                    # visualize_regions_queens(board, queens)

                    # Here we can do the symmetry check for when the color is on the
                    #   same row or column as the original queen
                    queen_loc_of_color = color_to_queen_loc[color]

                    if queen_loc_of_color[0] == row:
                        # Use the column to find the queen of that column
                        conflicting_queen_loc = None

                        for queen_loc in queens:
                            if queen_loc[1] == col:
                                conflicting_queen_loc = queen_loc
                                break

                        assert conflicting_queen_loc is not None  # TODO: remove

                        new_potential_current_queen_loc = (row,
                                                           col)
                        # Conflicting potential is current queen col, same row
                        new_potential_conflicting_queen_loc = (conflicting_queen_loc[0],
                                                               queen_loc_of_color[1])
                        
                        # Only do this check if the conflicting potential square uncolored
                        if board[new_potential_conflicting_queen_loc] == -1:
                            # Check immediate neighbors of both. If there is no neighboring
                            #   queen in either new potential spot then we have to mark
                            #   a forbidden color. Ignore the queen of the same color when looking
                            new_potential_current_queen_surroundings = \
                                get_adjacent_cells_diag(new_potential_current_queen_loc[0],
                                                        new_potential_current_queen_loc[1])
                            if queen_loc_of_color in new_potential_current_queen_surroundings:
                                new_potential_current_queen_surroundings.remove(queen_loc_of_color)

                            new_potential_conflicting_queen_surroundings = \
                                get_adjacent_cells_diag(new_potential_conflicting_queen_loc[0],
                                                        new_potential_conflicting_queen_loc[1])
                            if conflicting_queen_loc in new_potential_conflicting_queen_surroundings:
                                new_potential_conflicting_queen_surroundings.remove(conflicting_queen_loc)
                            
                            found_neighbor_queen_in_new_potential_loc = False
                            for q in queens:
                                if q in new_potential_current_queen_surroundings:
                                    # print("found neighbor queen for current potential", q)
                                    found_neighbor_queen_in_new_potential_loc = True
                                    break
                                if q in new_potential_conflicting_queen_surroundings:
                                    # print("found neighbor queen for conflicting potential", q)
                                    found_neighbor_queen_in_new_potential_loc = True
                                    break
                            
                            if not found_neighbor_queen_in_new_potential_loc:
                                square_to_disallowed_color_mapping[new_potential_conflicting_queen_loc].append(
                                    int(board[conflicting_queen_loc]))
                                # print(f"No neighbor conflicts found, need to mark illegal color of {board[conflicting_queen_loc]}")                        

                    elif queen_loc_of_color[1] == col:
                        # Use the row to find the queen of that row
                        conflicting_queen_loc = None

                        for queen_loc in queens:
                            if queen_loc[0] == row:
                                conflicting_queen_loc = queen_loc
                                break

                        assert conflicting_queen_loc is not None  # TODO: remove
                        # Current potential is where we will place new color
                        new_potential_current_queen_loc = (row,
                                                           col)
                        # Conflicting potential is current queen row, same col
                        new_potential_conflicting_queen_loc = (queen_loc_of_color[0],
                                                               conflicting_queen_loc[1])
                        # Only do this check if the conflicting potential square uncolored
                        if board[new_potential_conflicting_queen_loc] == -1:
                            # Check immediate neighbors of both. If there is no neighboring
                            #   queen in either new potential spot then we have to mark
                            #   a forbidden color. Ignore the queen of the same color when looking
                            new_potential_current_queen_surroundings = \
                                get_adjacent_cells_diag(new_potential_current_queen_loc[0],
                                                        new_potential_current_queen_loc[1])
                            new_potential_conflicting_queen_surroundings = \
                                get_adjacent_cells_diag(new_potential_conflicting_queen_loc[0],
                                                        new_potential_conflicting_queen_loc[1])

                            found_neighbor_queen_in_new_potential_loc = False
                            # ignore the swapping queens
                            queens_to_check = [q for q in queens
                                            if q != queen_loc_of_color
                                            if q != conflicting_queen_loc]
                            assert len(queens_to_check) == (n - 2)
                            for q in queens_to_check:
                                if q in new_potential_current_queen_surroundings:
                                    # print("found neighbor queen for current potential", q)
                                    found_neighbor_queen_in_new_potential_loc = True
                                    break
                                if q in new_potential_conflicting_queen_surroundings:
                                    # print("found neighbor queen for conflicting potential", q)
                                    found_neighbor_queen_in_new_potential_loc = True
                                    break
                            
                            if not found_neighbor_queen_in_new_potential_loc:
                                square_to_disallowed_color_mapping[new_potential_conflicting_queen_loc].append(
                                    int(board[conflicting_queen_loc]))
                                # print(f"No neighbor conflicts found, need to mark illegal color of {board[conflicting_queen_loc]}")

                else:  # Either 0 or 2 solutions, 0 probably can't happen since we start with valid queens
                    # Undo the color marking and remove from candidates
                    # print("REJECTED CANDIDATE")
                    board[row, col] = -1  # TODO: maybe just duplicate board if it gets messy?
                    candidates.remove((selected_score, (row, col, color)))
    
    return board


def get_adjacent_cells(row: int, col: int, n: int) -> List[Tuple[int, int]]:
    """Get orthogonally adjacent cells (no diagonals)"""
    adjacent = []
    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:  # up, down, left, right
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < n and 0 <= new_col < n:
            adjacent.append((new_row, new_col))
    return adjacent

def get_adjacent_colors(row: int, col: int, board: np.ndarray, n: int) -> Set[int]:
    """Get all colors adjacent to a cell"""
    colors = set()
    for adj_row, adj_col in get_adjacent_cells(row, col, n):
        color = board[adj_row, adj_col]
        if color != -1:
            colors.add(color)
    return colors


def visualize_regions_queens(board: np.ndarray, queens: List[Tuple[int, int]]):
    """Visualize the regions and queens"""
    n = board.shape[0]
    
    # Create color map with distinct colors
    num_colors = len(set(board.flatten()))
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, num_colors))
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot regions
    im = ax.imshow(board, cmap=plt.get_cmap('tab20'))
    
    # Plot queens
    queen_rows, queen_cols = zip(*queens)
    ax.scatter(queen_cols, queen_rows, color='red', s=300, marker='o', 
              edgecolor='white', linewidth=2, zorder=2, label='Queens')
    
    # Customize the plot
    ax.grid(True, color='black', linewidth=0.5)
    ax.set_xticks(np.arange(-0.5, n, 1))
    ax.set_yticks(np.arange(-0.5, n, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    plt.title(f"Queen Positions and Regions on {n}x{n} Board")
    plt.tight_layout()
    plt.show()
    # plt.show(block=False)


# Verify contiguity
def verify_regions(board: np.ndarray, queens: List[Tuple[int, int]]) -> bool:
    """
    Verify that:
    1. All cells are colored
    2. Each region is contiguous
    3. Each region contains exactly one queen
    """
    n = board.shape[0]
    num_regions = len(queens)
    
    def flood_fill(row: int, col: int, color: int, visited: Set[Tuple[int, int]]):
        """Flood fill to find all cells of the same color"""
        if (row, col) in visited or row < 0 or row >= n or col < 0 or col >= n or board[row, col] != color:
            return
        visited.add((row, col))
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            flood_fill(row + dr, col + dc, color, visited)
    
    # Check all cells are colored
    if -1 in board:
        print("Not all cells are colored")
        return False
    
    # Check each region
    for color in range(num_regions):
        # Find a starting cell for this color
        start = None
        queen_count = 0
        for i in range(n):
            for j in range(n):
                if board[i, j] == color:
                    if start is None:
                        start = (i, j)
                    if (i, j) in queens:
                        queen_count += 1
        
        # Check queen count
        if queen_count != 1:
            print(f"Region {color} has {queen_count} queens")
            return False
        
        # Check contiguity
        visited = set()
        flood_fill(start[0], start[1], color, visited)
        
        # Check if we found all cells of this color
        for i in range(n):
            for j in range(n):
                if board[i, j] == color and (i, j) not in visited:
                    print(f"Region {color} is not contiguous")
                    return False
    
    return True
def has_unique_solution(board: np.ndarray) -> bool:
    """
    Check if board has exactly one valid queen arrangement using forward checking and MRV.
    """
    n = board.shape[0]
    colors = sorted(set(board.ravel()))
    solution_count = 0
    
    # Pre-compute all positions for each color
    color_positions = {
        color: set((i, j) for i in range(n) for j in range(n) if board[i, j] == color)
        for color in colors
    }
    
    class State:
        """Class to track the current state and available moves"""
        def __init__(self):
            self.used_rows = set()
            self.used_cols = set()
            self.used_positions = set()
            self.placed_colors = set()
            # For each color, maintain set of currently valid positions
            self.available_moves = {
                color: color_positions[color].copy() 
                for color in colors
            }
        
        def place_queen(self, pos: Tuple[int, int], color: int) -> Set[Tuple[int, int]]:
            """Place a queen and return affected positions"""
            row, col = pos
            affected = set()
            
            # Mark position and its row/col as used
            self.used_positions.add(pos)
            self.used_rows.add(row)
            self.used_cols.add(col)
            self.placed_colors.add(color)
            
            # Remove this position from all available moves
            for moves in self.available_moves.values():
                if pos in moves:
                    moves.remove(pos)
            
            # Remove all positions in same row/col/adjacent
            for c in range(n):
                affected.add((row, c))
            for r in range(n):
                affected.add((r, col))
            for r in range(max(0, row-1), min(n, row+2)):
                for c in range(max(0, col-1), min(n, col+2)):
                    affected.add((r, c))
            
            # Remove affected positions from available moves
            for moves in self.available_moves.values():
                moves.difference_update(affected)
                
            return affected
        
        def undo_move(self, pos: Tuple[int, int], color: int, affected: Set[Tuple[int, int]]):
            """Undo a queen placement"""
            row, col = pos
            self.used_positions.remove(pos)
            self.used_rows.remove(row)
            self.used_cols.remove(col)
            self.placed_colors.remove(color)
            
            # Restore available moves for all colors
            for c in colors:
                self.available_moves[c] = {
                    pos for pos in color_positions[c]
                    if pos not in self.used_positions
                    and pos[0] not in self.used_rows
                    and pos[1] not in self.used_cols
                    and not any((abs(pos[0] - r) <= 1 and abs(pos[1] - c) <= 1)
                               for r, c in self.used_positions)
                }
    
        def get_next_color(self) -> Optional[int]:
            """Choose next color using MRV heuristic"""
            remaining = [(c, len(moves)) for c, moves in self.available_moves.items() 
                        if moves and c not in self.placed_colors]
            return min(remaining, key=lambda x: x[1])[0] if remaining else None
    
    def backtrack(state: State) -> None:
        nonlocal solution_count
        
        if solution_count > 1:
            return
            
        if len(state.placed_colors) == len(colors):
            solution_count += 1
            return
        
        # Use MRV to choose next color
        color = state.get_next_color()
        if color is None:
            return
            
        # Try each valid position for this color
        positions = sorted(state.available_moves[color],
                         key=lambda pos: len([1 for r in range(max(0, pos[0]-1), min(n, pos[0]+2))
                                            for c in range(max(0, pos[1]-1), min(n, pos[1]+2))
                                            if (r, c) in state.used_positions]))
        
        for pos in positions:
            # Forward checking: place queen and track affected positions
            affected = state.place_queen(pos, color)
            
            # If any color has no valid moves left, skip this branch
            if all(len(moves) > 0 or c in state.placed_colors 
                   for c, moves in state.available_moves.items()):
                backtrack(state)
            
            # Undo move
            state.undo_move(pos, color, affected)
            
            if solution_count > 1:
                return
    
    # Start search
    initial_state = State()
    backtrack(initial_state)
    
    return solution_count == 1



def find_unique_solution_board(n: int, max_attempts: int = 1000000) -> Optional[Tuple[np.ndarray, int]]:
    """
    Optimized version of board finder.
    """
    start_time = time.time()
    
    # Generate initial queens only once (?)
    # queens = generate_random_queens(n)
    
    # Keep track of seen boards to avoid duplicates
    seen_boards = set()
    
    for attempt_num in range(max_attempts):
        # board = generate_regions(queens, n)    
        queens = generate_random_queens(n)
        # visualize_queens(queens, n=n)

        # board = generate_regions_optimized(queens, n)
        board = generate_regions_jagged(queens, n)

        if (attempt_num+1) % 10 == 0:
            elapsed = time.time() - start_time
            boards_per_sec = attempt_num / elapsed if elapsed > 0 else 0
            print(f"Attempt {attempt_num}, {boards_per_sec:.1f} boards/sec")            
        
        # TODO: incorporate queen placement into this
        # board_hash = hash(board.tobytes())
        # if board_hash in seen_boards:
        #     continue
            
        # seen_boards.add(board_hash)

        if board is None:
            # We ended up in a state where unique solution was impossible
            continue

        if has_unique_solution(board):
            print(f"Found unique solution board after {attempt_num} attempts")
            attempt_time = time.time() - start_time
            print(f"Took {attempt_time:.2f} seconds")

            print(board)
            print(queens)
            # visualize_regions_queens(board, queens)
            return board, attempt_num, attempt_time
        else:
            pass
            # solutions = find_up_to_two_solutions(board)
            
            # solution_one = [(int(r), int(c)) for r,c in solutions[0]]
            # solution_two = [(int(r), int(c)) for r,c in solutions[1]]
            # visualize_regions_queens(board, solution_one)
            # visualize_regions_queens(board, solution_two)
            # import pdb; pdb.set_trace()

    return None

def visualize_regions(board: np.ndarray):
    """Visualize the regions and queens"""
    n = board.shape[0]
    
    # Create color map with distinct colors
    num_colors = len(set(board.flatten()))
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, num_colors))
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot regions
    im = ax.imshow(board, cmap=plt.get_cmap('tab20'))
        
    # Customize the plot
    ax.grid(True, color='black', linewidth=0.5)
    ax.set_xticks(np.arange(-0.5, n, 1))
    ax.set_yticks(np.arange(-0.5, n, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    plt.title(f"Queen Positions and Regions on {n}x{n} Board")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate puzzle boards with unique queen solutions')
    
    # Add arguments
    parser.add_argument('--size', '-n', 
                       type=int, 
                       default=8,
                       help='Size of the board (default: 8)')
    args = parser.parse_args()

    n = args.size

    # # Example usage
    # n = 8
    # queens = generate_random_queens(n)
    # print(f"Queens: {queens}")
    # board = generate_regions(queens, n)

    # board = np.array([
    #     [0, 0, 0, 2, 4, 4, 4, 4, 4, 4, 4, 4],
    #     [0, 0, 2, 2, 3, 3, 3, 4, 4, 3, 5, 4],
    #     [0, 0, 2, 2, 3, 3, 4, 4, 4, 3, 5, 5],
    #     [1, 0, 2, 2, 2, 3, 3, 3, 3, 3, 5, 6],
    #     [1, 0, 1, 2, 2, 2, 3, 3, 5, 5, 5, 6],
    #     [1, 1, 1, 1, 1, 2, 3, 5, 5, 5, 6, 6],
    #     [3, 3, 3, 3, 3, 3, 3, 3, 10, 10, 10, 6],
    #     [3, 7, 7, 7, 7, 3, 3, 10, 10, 10, 10, 6],
    #     [3, 7, 3, 7, 3, 3, 3, 3, 10, 3, 10, 11],
    #     [3, 3, 3, 7, 3, `8`, 8, 3, 3, 3, 3, 11],
    #     [3, 3, 3, 7, 3, 8, 8, 9, 9, 9, 9, 9],
    #     [3, 3, 3, 7, 8, 8, 9, 9, 9, 9, 9, 9]
    # ])

    # print(has_unique_solution(board))

    times = []
    num_attempts = []

    num_trials = 10
    for i in range(num_trials):
        print(i)
        board, attempt_num, attempt_time = find_unique_solution_board(n)
        times.append(attempt_time)
        num_attempts.append(attempt_num)
        # print(i, has_unique_solution(board))

    print(f"Average time per find: {sum(times) / len(times)}")
    print(f"Average number of attempts: {sum(num_attempts) / len(num_attempts)}")

    # solutions = find_up_to_two_solutions(board)
    # if len(solutions) == 0:
    #     print("No solutions found!")
    # elif len(solutions) == 1:
    #     print("Unique solution found!")
    #     print("Queen positions:", [(int(r), int(c)) for r,c in solutions[0]])
    # else:
    #     print("Multiple solutions exist!")
    #     print("First solution:", [(int(r), int(c)) for r,c in solutions[0]])
        # print("Second solution:", [(int(r), int(c)) for r,c in solutions[1]])


    # # Verify the solution
    # print("Regions are valid:", verify_regions(board, queens))

    # visualize_regions(board)
