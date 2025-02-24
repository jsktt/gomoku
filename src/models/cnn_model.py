# src/models/cnn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import BaseModel

class GomokuCNN(nn.Module, BaseModel):
    def __init__(self, board_size=15):
        nn.Module.__init__(self)
        BaseModel.__init__(self)
        
        self.board_size = board_size
        
        # Neural network
        self.conv1 = nn.Conv2d(2, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Policy head
        self.policy_conv = nn.Conv2d(256, 2, 1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # Value head
        self.value_conv = nn.Conv2d(256, 1, 1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.to(self.device)
    
    def forward(self, x):
        # Common layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Policy head
        policy = self.policy_conv(x)
        policy = policy.view(-1, 2 * self.board_size * self.board_size)
        policy = self.policy_fc(policy)
        policy = F.softmax(policy, dim=1)
        
        # Value head
        value = self.value_conv(x)
        value = value.view(-1, self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

    def get_move(self, board_state):
        """Get next move given current board state."""
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            state_tensor = self._preprocess_state(board_state)
            policy, _ = self(state_tensor)
            
            # Convert policy to move probabilities
            move_probs = policy.squeeze().cpu().numpy()
            
            # Mask invalid moves
            valid_moves = board_state == 0
            move_probs = move_probs.reshape(self.board_size, self.board_size)
            move_probs[~valid_moves] = -float('inf')
            
            # Select move with highest probability
            move = np.unravel_index(move_probs.argmax(), move_probs.shape)
            return move

    def train_network(self, states, policies, values):
        """Training method for the network."""
        # Data is already in tensor format and on the correct device
    
        # Forward pass
        pred_policies, pred_values = self(states)
    
        # Calculate losses
        policy_loss = F.cross_entropy(pred_policies, policies)
        value_loss = F.mse_loss(pred_values.squeeze(), values)
        total_loss = policy_loss + value_loss
    
        # Backward pass and optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
    
        return total_loss.item()

    def save(self, path: str):
        """Save model parameters."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load model parameters."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def _preprocess_state(self, state):
        """Convert numpy board state to torch tensor."""
        black = (state == 1).astype(np.float32)
        white = (state == -1).astype(np.float32)
        state_tensor = torch.FloatTensor(np.stack([black, white]))
        return state_tensor.unsqueeze(0).to(self.device)