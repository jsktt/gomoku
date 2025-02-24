import numpy as np

class Board:
    def __init__(self, size=15):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1  # 1 for black, -1 for white
        self.last_move = None
    
    def make_move(self, x: int, y: int) -> bool:
        """Make a move on the board."""
        if self.is_valid_move(x, y):
            self.board[x, y] = self.current_player
            self.last_move = (x, y)
            self.current_player *= -1
            return True
        return False
    
    def is_valid_move(self, x: int, y: int) -> bool:
        """Check if move is valid."""
        return 0 <= x < self.size and 0 <= y < self.size and self.board[x, y] == 0
    
    def check_win(self, x: int, y: int) -> bool:
        """Check if the last move at (x,y) won the game."""
        if not self.last_move:
            return False
            
        player = self.board[x, y]
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        
        for dx, dy in directions:
            count = 1
            # Check both directions
            for sign in [1, -1]:
                for i in range(1, 5):
                    nx, ny = x + sign*i*dx, y + sign*i*dy
                    if not (0 <= nx < self.size and 0 <= ny < self.size):
                        break
                    if self.board[nx, ny] != player:
                        break
                    count += 1
            if count >= 5:
                return True
        return False
    
    def get_state(self):
        """Get the current board state."""
        return self.board.copy()
    
    def get_valid_moves(self):
        """Get list of valid moves."""
        return [(i, j) for i in range(self.size) 
                for j in range(self.size) if self.board[i, j] == 0]
    
    def __str__(self):
        symbols = {0: '.', 1: '●', -1: '○'}
        board_str = '   ' + ' '.join(f'{i:2}' for i in range(self.size)) + '\n'
        for i in range(self.size):
            board_str += f'{i:2} ' + ' '.join(symbols[self.board[i, j]] 
                        for j in range(self.size)) + '\n'
        return board_str