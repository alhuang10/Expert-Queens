from board_generator import find_unique_solution_board
from get_solutions import find_up_to_two_solutions

def test_small_board_generation():
    # Test generations from 6 through 9, quick sanity check
    for size in [6,7,8,9]:
        board, _ = find_unique_solution_board(n=size, max_attempts=10)
        solutions = find_up_to_two_solutions(board)

        assert board is not None, f"Failed at finding board of size {size}"
        assert len(solutions) == 1, f"Found a multiple solution board"


def test_10_by_10_generation():
    board, _ = find_unique_solution_board(n=10, max_attempts=10)
    assert board is not None, f"Failed at finding 10 by 10 board"
