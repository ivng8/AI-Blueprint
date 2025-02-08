class TicTacToe:
    def __init__(self):
        self.board = [['', '', ''], ['', '', ''], ['', '', '']]
        self.turn = 'X'

    def move(self, row, col):
        if self.board[row][col] == '':
            self.board[row][col] = self.turn
            if self.turn == 'X':
                self.turn == 'O'
            else:
                self.turn == 'X'

    def utility(self):
        winner = ''
        if self.board[0][0] == self.board[1][1] == self.board[2][2]:
            winner = self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0]:
            winner = self.board[0][2]
        
        match winner:
            case 'X':
                return 1
            case 'O':
                return -1
            case '':
                return 0
            case _:
                return None