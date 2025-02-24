import numpy as np
import math
from typing import List, Tuple, Dict
from ..core.game import Game
from .base_model import BaseModel

class MCTSNode:
    def __init__(self, game_state: np.ndarray, parent=None, prior_p=1.0):
        self.game_state = game_state
        self.parent = parent
        self.children: Dict[Tuple[int, int], MCTSNode] = {}
        self.n_visits = 0
        self.Q = 0  # mean action value
        self.P = prior_p  # prior probability
        
    def expand(self, game: Game):
        """Expand the node with all possible moves."""
        valid_moves = game.get_valid_moves()
        
        for move in valid_moves:
            if move not in self.children:
                new_game = Game()
                new_game.board.board = self.game_state.copy()
                new_game.make_move(*move)
                self.children[move] = MCTSNode(
                    new_game.board.board,
                    parent=self,
                    prior_p=1.0 / len(valid_moves)  # uniform prior
                )

    def get_value(self, c_puct: float) -> float:
        """Calculate node's value with UCB formula."""
        U = c_puct * self.P * math.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self.Q + U

    def update(self, value: float):
        """Update node values from leaf evaluation."""
        self.n_visits += 1
        self.Q += (value - self.Q) / self.n_visits

    def is_leaf(self) -> bool:
        """Check if node is leaf."""
        return len(self.children) == 0

class MCTS(BaseModel):
    def __init__(self, n_simulations=100, c_puct=5):
        self.n_simulations = n_simulations
        self.c_puct = c_puct

    def _playout(self, node: MCTSNode, game: Game):
        """Execute one playout from the root to the leaf."""
        # Selection
        while not node.is_leaf():
            # Find child that maximizes UCB
            move, node = max(
                node.children.items(),
                key=lambda item: item[1].get_value(self.c_puct)
            )
            game.make_move(*move)

        # Expansion
        if not game.is_game_over():
            node.expand(game)

        # Simulation
        value = self._simulate(game)

        # Backpropagation
        while node is not None:
            node.update(value)
            value = -value  # Switch perspective
            node = node.parent

    def _simulate(self, game: Game) -> float:
        """Run a simulation from current state."""
        current_player = game.board.current_player
        
        while not game.is_game_over():
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                return 0  # Draw
            # Random playout
            move = valid_moves[np.random.randint(len(valid_moves))]
            game.make_move(*move)

        winner = game.get_winner()
        if winner == 0:
            return 0
        return 1 if winner == current_player else -1

    def get_move(self, board_state: np.ndarray) -> Tuple[int, int]:
        """Get best move using MCTS."""
        root = MCTSNode(board_state)
        game = Game()
        game.board.board = board_state.copy()

        for _ in range(self.n_simulations):
            game_copy = Game()
            game_copy.board.board = board_state.copy()
            self._playout(root, game_copy)

        # Select move with most visits
        return max(root.children.items(),
                  key=lambda item: item[1].n_visits)[0]

    def train(self, states, actions, rewards):
        """Training is not applicable for pure MCTS."""
        pass

    def save(self, path: str):
        """Save is not applicable for pure MCTS."""
        pass

    def load(self, path: str):
        """Load is not applicable for pure MCTS."""
        pass