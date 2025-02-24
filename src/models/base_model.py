from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    @abstractmethod
    def get_move(self, board_state: np.ndarray) -> tuple:
        """Get next move given current board state."""
        pass
    
    @abstractmethod
    def train(self, states, actions, rewards):
        """Train the model."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save model to path."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load model from path."""
        pass
