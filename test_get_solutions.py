import numpy as np

from get_solutions import find_up_to_two_solutions, find_up_to_two_solutions_optimized


def test_unique_solution_board_big():
    unique_solution_board = np.array([
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
    # solutions = find_up_to_two_solutions(unique_solution_board)
    solutions = find_up_to_two_solutions_optimized(unique_solution_board)
    assert len(solutions) == 1

def test_unique_solution_board_small():
    unique_solution_board = np.array([
        [6, 6, 6, 6, 6, 6, 6, 4],
        [6, 2, 5, 5, 0, 0, 6, 6],
        [2, 2, 1, 1, 1, 0, 0, 6],
        [2, 1, 1, 1, 1, 6, 6, 6],
        [2, 2, 1, 1, 6, 6, 3, 3],
        [1, 2, 1, 6, 6, 6, 6, 6],
        [1, 1, 1, 1, 6, 6, 6, 6],
        [1, 7, 1, 6, 6, 6, 6, 6]
    ])
    # solutions = find_up_to_two_solutions(unique_solution_board)
    solutions = find_up_to_two_solutions_optimized(unique_solution_board)
    assert len(solutions) == 1


def test_non_unique_solution_board_small():
    non_unique_solution_board = np.array([
       [-1, -1, -1,  3,  3, -1],
       [-1, -1, -1, -1, -1,  1],
       [ 4, -1,  0, -1, -1, -1],
       [ 4, -1,  0, -1, -1, -1],
       [-1, -1, -1, -1,  2, -1],
       [-1,  5, -1, -1, -1, -1]])
    solutions = find_up_to_two_solutions_optimized(non_unique_solution_board)
    assert len(solutions) == 2

def test_non_unique_solution_board_big():
    non_unique_solution_board = np.array([
       [ 7,  7,  9,  8, 10, 10, 10, 10,  4,  4,  4,  4],
       [ 7,  7,  9,  8, 10, 10, 10, 10, 10,  4,  4,  4],
       [ 7,  8,  9,  8,  8, 10, 10, 10, 10, 10,  4,  4],
       [ 7,  8,  8,  8, 10, 10, 10, 10, 10, 10, 10, 10],
       [ 7,  7,  7,  7, 10,  6, 10, 10, 10, 10, 10,  1],
       [ 7,  7,  7,  7, 10,  6,  6,  5,  5,  1,  1,  1],
       [ 2,  7,  7,  7, 10,  2,  6,  6,  6,  1,  1,  1],
       [ 2,  2,  2, 11,  2,  2,  6,  1,  1,  1,  1,  1],
       [ 2,  2,  2,  2,  2,  6,  6,  1,  2,  1,  1,  1],
       [ 2,  2,  2,  2,  6,  6,  6,  1,  2,  1,  0,  3],
       [ 2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3],
       [ 2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3]])

    solutions = find_up_to_two_solutions_optimized(non_unique_solution_board)
    assert len(solutions) == 2