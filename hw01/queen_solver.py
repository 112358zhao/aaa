"""
Eight Queens Problem Solver

This module provides multiple approaches to solving the classic Eight Queens problem, which involves placing eight queens on an 8x8 chessboard such that no two queens threaten each other.

Approaches implemented:
1. Backtracking
2. Set-based approach using Python sets

Usage:
To use the solver, simply import the module and call the appropriate function.

Example:
>>> from queen_solver import backtracking_solver
>>> solutions = backtracking_solver()

This will return a list of all possible solutions, where each solution is represented as a list of integers. Each integer represents the column index for a queen's position in the corresponding row.
"""

def backtracking_solver():
    solutions = []
    board = [-1] * 8  # board[i] = column index of the queen in row i
    solve(0, board, solutions)
    return solutions


def solve(row, board, solutions):
    if row == 8:
        solutions.append(board[:])  # Found a solution
        return
    for col in range(8):
        if is_safe(row, col, board):
            board[row] = col
            solve(row + 1, board, solutions)
            board[row] = -1  # Reset the position


def is_safe(row, col, board):
    for r in range(row):
        c = board[r]
        if c == col or abs(c - col) == abs(r - row):
            return False
    return True


def set_based_solver():
    solutions = []
    col_set = set()
    diag_set1 = set()
    diag_set2 = set()
    board = [-1] * 8
    solve_set_based(0, board, col_set, diag_set1, diag_set2, solutions)
    return solutions


def solve_set_based(row, board, col_set, diag_set1, diag_set2, solutions):
    if row == 8:
        solutions.append(board[:])
        return
    for col in range(8):
        if col not in col_set and (row - col) not in diag_set1 and (row + col) not in diag_set2:
            board[row] = col
            col_set.add(col)
            diag_set1.add(row - col)
            diag_set2.add(row + col)
            solve_set_based(row + 1, board, col_set, diag_set1, diag_set2, solutions)
            board[row] = -1
            col_set.remove(col)
            diag_set1.remove(row - col)
            diag_set2.remove(row + col)
