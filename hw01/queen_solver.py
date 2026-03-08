def backtracking_solver(n):
    board = [[0] * n for _ in range(n)]
    if solve_n_queens(board, 0, n):
        return board
    return None  # No solution found


def solve_n_queens(board, row, n):
    if row >= n:
        return True
    for col in range(n):
        if is_safe(board, row, col, n):
            board[row][col] = 1  # Place queen
            if solve_n_queens(board, row + 1, n):
                return True
            board[row][col] = 0  # Backtrack
    return False


def is_safe(board, row, col, n):
    for i in range(row):
        if board[i][col] == 1:
            return False  # Check column
    for i, j in zip(range(row - 1, -1, -1), range(col - 1, -1, -1)):
        if board[i][j] == 1:
            return False  # Check upper left diagonal
    for i, j in zip(range(row - 1, -1, -1), range(col + 1, n)):
        if board[i][j] == 1:
            return False  # Check upper right diagonal
    return True



def set_based_solver(n):
    col_set = set()
    pos_diag_set = set()
    neg_diag_set = set()
    board = [[0] * n for _ in range(n)]
    if solve_with_sets(board, 0, col_set, pos_diag_set, neg_diag_set, n):
        return board
    return None  # No solution found


def solve_with_sets(board, row, col_set, pos_diag_set, neg_diag_set, n):
    if row >= n:
        return True
    for col in range(n):
        if col not in col_set and (row + col) not in pos_diag_set and (row - col) not in neg_diag_set:
            board[row][col] = 1  # Place queen
            col_set.add(col)
            pos_diag_set.add(row + col)
            neg_diag_set.add(row - col)
            if solve_with_sets(board, row + 1, col_set, pos_diag_set, neg_diag_set, n):
                return True
            board[row][col] = 0  # Backtrack
            col_set.remove(col)
            pos_diag_set.remove(row + col)
            neg_diag_set.remove(row - col)
    return False

# Existing functionality for 8-queens

if __name__ == '__main__':
    n = 8  # You can change this value for different N
    print("Backtracking Solver:")
    result = backtracking_solver(n)
    if result:
        for row in result:
            print(row)
    else:
        print("No solution found.")
    
    print("Set-Based Solver:")
    result = set_based_solver(n)
    if result:
        for row in result:
            print(row)
    else:
        print("No solution found.")