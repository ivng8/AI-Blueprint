import unittest 

class TicTacToe:
    def __init__(self):
        self.board = [['', '', ''], ['', '', ''], ['', '', '']]
        self.turn = 'X'

    def move(self, row, col):
        if self.board[row][col] == '':
            self.board[row][col] = self.turn
            if self.turn == 'X':
                self.turn = 'O'
            else:
                self.turn = 'X'

    def utility(self):
        winner = ''
        if self.board[0][0] == self.board[1][1] == self.board[2][2]:
            winner = self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0]:
            winner = self.board[0][2]
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] and self.board[i][0]:
                winner = self.board[i][0]
            if self.board[0][i] == self.board[1][i] == self.board[2][i] and self.board[i][0]:
                winner = self.board[0][i]
        if winner == 'X':
            return 1
        elif winner == 'O':
            return -1
        if all(cell != '' for row in self.board for cell in row):
            return 0
        else:
            return None
        
    def minimax(self): # doesn't need player parameter because of self bookkeeping
        if self.utility() is None:
            best_score = float("inf")
            if self.turn == 'X':
                best_score = best_score * -1
            for x in range(3):
                for y in range(3):
                    if self.board[x][y] == '':
                        self.move(x, y)
                    
                        score = self.minimax()

                        self.board[x][y] = ''
                        if self.turn == 'X':
                            self.turn = 'O'
                        else:
                            self.turn = 'X'

                        if self.turn == 'X':
                            best_score = max(best_score, score)
                        else:
                            best_score = min(best_score, score)
            return best_score
        else:
            return self.utility()
        
"""
The move in s2 is suboptimal.
The only optimal move would be place the O in the middle.
The move in s3 is suboptimal.
The alternative moves that are optimal would be to place the X
in either the top right or bottom left corner.
The move in s4 is suboptimal.
The alternative moves that are optimal would be to place the O
in the top right, middle, middle right, or bottom middle.
"""        

class TestTicTacToe(unittest.TestCase):

    def setUp(self):
        """Set up a fresh TicTacToe instance before each test."""
        self.game = TicTacToe()

    def test_move_valid(self):
        """Test valid moves on the board."""
        self.game.move(0, 0)
        self.assertEqual(self.game.board[0][0], 'X')
        self.assertEqual(self.game.turn, 'O')  # Check player switch

        self.game.move(1, 1)
        self.assertEqual(self.game.board[1][1], 'O')
        self.assertEqual(self.game.turn, 'X')  # Check player switch

    def test_move_invalid(self):
        """Test moves on occupied cells."""
        self.game.move(0, 0)  # X moves
        self.game.move(0, 0)  # O tries same move
        self.assertEqual(self.game.board[0][0], 'X')  # Board should stay the same
        self.assertEqual(self.game.turn, 'O')  # Player should still be O

    def test_utility_win_X(self):
        """Test utility function when X wins."""
        self.game.board = [['X', 'X', 'X'], ['', '', ''], ['', '', '']]
        self.assertEqual(self.game.utility(), 1)

    def test_utility_win_O(self):
        """Test utility function when O wins."""
        self.game.board = [['O', 'O', 'O'], ['', '', ''], ['', '', '']]
        self.assertEqual(self.game.utility(), -1)

    def test_utility_draw(self):
        """Test utility function for a draw."""
        self.game.board = [['X', 'O', 'X'], ['X', 'O', 'O'], ['O', 'X', 'X']]
        self.assertEqual(self.game.utility(), 0)

    def test_utility_ongoing(self):
        """Test utility function when the game is ongoing."""
        self.game.board = [['X', '', ''], ['O', '', ''], ['', '', '']]
        self.assertIsNone(self.game.utility())

    def test_minimax_empty_board(self):
        """Minimax should return 0 for an empty board since it's a neutral state."""
        result = self.game.minimax()
        self.assertEqual(result, 0, "Minimax should return 0 for an empty board.")

    def test_minimax_win_X(self):
        """Minimax should return 1 if X is guaranteed a win."""
        self.game.board = [['X', 'X', ''],
                            ['O', 'O', ''],
                            ['', '', '']]
        self.game.turn = 'X'
        result = self.game.minimax()
        self.assertEqual(result, 1, "Minimax should return 1 if X can win.")

    def test_minimax_win_O(self):
        """Minimax should return -1 if O is guaranteed a win."""
        self.game.board = [['O', 'O', ''],
                            ['X', 'X', ''],
                            ['', '', '']]
        self.game.turn = 'O'
        result = self.game.minimax()
        self.assertEqual(result, -1, "Minimax should return -1 if O can win.")

    def test_minimax_draw(self):
        """Minimax should return 0 if the board results in a draw."""
        self.game.board = [['X', 'O', 'X'],
                            ['X', 'X', 'O'],
                            ['O', 'X', 'O']]
        self.game.turn = 'X'  # or 'O'; it doesn't matter for a draw
        result = self.game.minimax()
        self.assertEqual(result, 0, "Minimax should return 0 for a draw state.")
    
    def test_minimax_almost_full(self):
        """Minimax should make the best move in an almost-full board."""
        self.game.board = [['X', 'O', 'X'],
                            ['X', 'X', 'O'],
                            ['O', 'X', '']]
        self.game.turn = 'O'
        result = self.game.minimax()
        self.assertEqual(result, 0, "Minimax should return 0 in this forced-draw state.")

    def test_ex6(self):
        self.game.board = [['O', 'O', 'X'],
                            ['X', '', 'O'],
                            ['', '', 'X']]
        self.game.turn = 'X'
        result = self.game.minimax()
        self.assertEqual(result, 1)

    def test_ex6s1(self):
        self.game.board = [['', '', ''],
                            ['', '', ''],
                            ['', '', 'X']]
        self.game.turn = 'O'
        result = self.game.minimax()
        self.assertEqual(result, 0)

    def test_ex6s2(self):
        self.game.board = [['O', '', ''],
                            ['', '', ''],
                            ['', '', 'X']]
        self.game.turn = 'X'
        result = self.game.minimax()
        self.assertEqual(result, 1)

    def test_ex6s3(self):
        self.game.board = [['O', '', ''],
                            ['X', '', ''],
                            ['', '', 'X']]
        self.game.turn = 'O'
        result = self.game.minimax()
        self.assertEqual(result, 0)

    def test_ex6s4(self):
        self.game.board = [['O', 'O', ''],
                            ['X', '', ''],
                            ['', '', 'X']]
        self.game.turn = 'X'
        result = self.game.minimax()
        self.assertEqual(result, 1)

    def test_ex6s5(self):
        self.game.board = [['O', 'O', 'X'],
                            ['X', '', ''],
                            ['', '', 'X']]
        self.game.turn = 'O'
        result = self.game.minimax()
        self.assertEqual(result, 1)

if __name__ == '__main__':
    unittest.main()