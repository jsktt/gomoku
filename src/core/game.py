from typing import Optional, Tuple
from .board import Board
import numpy as np

class Game:
    def __init__(self, board_size: int = 15):
        self.board = Board(board_size)
        self.game_over = False
        self.winner = None
    
    def make_move(self, x: int, y: int) -> bool:
        """Make a move and return whether it was successful."""
        if self.game_over:
            return False
            
        if self.board.make_move(x, y):
            if self.board.check_win(x, y):
                self.game_over = True
                self.winner = -self.board.current_player  # Winner is the player who just moved
            elif not self.board.get_valid_moves():
                self.game_over = True
                self.winner = 0  # Draw
            return True
        return False
    
    def get_state(self) -> np.ndarray:
        """Get current game state."""
        return self.board.get_state()
    
    def get_valid_moves(self) -> list:
        """Get list of valid moves."""
        return self.board.get_valid_moves()
    
    def is_game_over(self) -> bool:
        """Check if game is over."""
        return self.game_over
    
    def get_winner(self) -> Optional[int]:
        """Get winner of the game (1 for black, -1 for white, 0 for draw, None if game not over)."""
        return self.winner if self.game_over else None