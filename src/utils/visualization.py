import matplotlib.pyplot as plt
import numpy as np

def plot_board(board: np.ndarray):
    """Plot the current board state."""
    plt.figure(figsize=(10, 10))
    plt.imshow(board, cmap='RdBu')
    plt.grid(True)
    plt.xticks(range(len(board)))
    plt.yticks(range(len(board)))
    plt.show()

def plot_training_history(history):
    """Plot training metrics."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'])
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()