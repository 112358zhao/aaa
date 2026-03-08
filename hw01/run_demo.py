def solve_n_queens(n):
    def is_safe(board, row, col):
        for i in range(col):
            if board[row][i] == 1:
                return False
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if board[i][j] == 1:
                return False
        for i, j in zip(range(row, n), range(col, -1, -1)):
            if board[i][j] == 1:
                return False
        return True

    def solve_util(board, col):
        if col >= n:
            return True
        for i in range(n):
            if is_safe(board, i, col):
                board[i][col] = 1
                if solve_util(board, col + 1):
                    return True
                board[i][col] = 0
        return False

    board = [[0] * n for _ in range(n)]
    if not solve_util(board, 0):
        return []
    return board


if __name__ == '__main__':
    for n in [4, 8]:
        solution = solve_n_queens(n)
        print(f'Solution for N={n}:')
        if solution:
            for row in solution:
                print(' '.join('Q' if x == 1 else '.' for x in row))
        else:
            print('No solution found.')