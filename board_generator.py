import random
from typing import List, Set, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np

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
            return result


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
    Returns an nxn numpy array where each cell contains a number 0 to n-1 representing its region.
    """
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
    
    return board

def visualize_regions(board: np.ndarray, queens: List[Tuple[int, int]]):
    """Visualize the regions and queens"""
    n = board.shape[0]
    
    # Create color map with distinct colors
    num_colors = len(set(board.flatten()))
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, num_colors))
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot regions
    im = ax.imshow(board, cmap=plt.cm.get_cmap('tab20'))
    
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

# # Example usage
# n = 12
# queens = generate_random_queens(n)
# print(f"Queens: {queens}")
# board = generate_regions(queens, n)
# print("Board:", board)

# # Verify the solution
# print("Regions are valid:", verify_regions(board, queens))

# # visualize_regions(board, queens)
