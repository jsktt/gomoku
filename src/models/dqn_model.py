# src/models/dqn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from .base_model import BaseModel

class DQN(nn.Module):
    def __init__(self, board_size=15):
        super(DQN, self).__init__()
        self.board_size = board_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Calculate size after convolutions
        # After conv layers, the spatial dimensions remain the same (15x15)
        # but we have 128 channels
        self.conv_out_size = 128 * board_size * board_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_out_size, 512)
        self.fc2 = nn.Linear(512, board_size * board_size)  # Output for each possible move

    def forward(self, x):
        batch_size = x.size(0)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))  # Output: batch_size x 64 x 15 x 15
        x = F.relu(self.conv2(x))  # Output: batch_size x 128 x 15 x 15
        x = F.relu(self.conv3(x))  # Output: batch_size x 128 x 15 x 15
        
        # Flatten
        x = x.view(batch_size, -1)  # Output: batch_size x (128*15*15)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))     # Output: batch_size x 512
        x = self.fc2(x)             # Output: batch_size x 225 (15*15)
        
        return x

class DQNAgent(BaseModel):
    def __init__(self, board_size=15, memory_size=10000, batch_size=32, 
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 learning_rate=0.001):
        """
        Initialize DQN Agent.
        
        Args:
            board_size (int): Size of the Gomoku board
            memory_size (int): Size of replay memory
            batch_size (int): Size of training batch
            gamma (float): Discount factor
            epsilon (float): Initial exploration rate
            epsilon_min (float): Minimum exploration rate
            epsilon_decay (float): Decay rate for exploration
            learning_rate (float): Learning rate for optimizer
        """
        self.board_size = board_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize networks
        self.policy_net = DQN(board_size).to(self.device)
        self.target_net = DQN(board_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def get_move(self, board_state):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            board_state (numpy.ndarray): Current board state
            
        Returns:
            tuple: (x, y) coordinates of the selected move
        """
        if random.random() < self.epsilon:
            # Random move
            valid_moves = [(i, j) for i in range(self.board_size) 
                          for j in range(self.board_size) if board_state[i, j] == 0]
            return random.choice(valid_moves) if valid_moves else None
        
        # Greedy move
        state_tensor = self._preprocess_state(board_state)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze()
            
        # Reshape to board dimensions
        q_values = q_values.view(self.board_size, self.board_size)
        
        # Mask invalid moves
        valid_moves = (board_state == 0)
        q_values[~valid_moves] = float('-inf')
        
        # Get best valid move
        move = np.unravel_index(torch.argmax(q_values.cpu()).item(), (self.board_size, self.board_size))
        return move

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))

    def train(self, states=None, actions=None, rewards=None):
        """
        Train the DQN on a batch from replay memory.
        
        Returns:
            float: Loss value
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors and move to correct device
        state_batch = torch.stack([self._preprocess_state(s).squeeze(0) for s in states]).to(self.device)
        action_batch = torch.tensor([(a[0] * self.board_size + a[1]) for a in actions], 
                                  device=self.device, dtype=torch.long)
        reward_batch = torch.tensor(rewards, device=self.device, dtype=torch.float)
        next_state_batch = torch.stack([self._preprocess_state(s).squeeze(0) for s in next_states]).to(self.device)
        done_batch = torch.tensor(dones, device=self.device, dtype=torch.float)
        
        # Compute Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = F.smooth_l1_loss(current_q_values.squeeze(), expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()

    def update_target_network(self):
        """Update target network parameters."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _preprocess_state(self, state):
        """
        Convert numpy board state to torch tensor.
        
        Args:
            state (numpy.ndarray): Board state
            
        Returns:
            torch.Tensor: Preprocessed state tensor
        """
        # Add batch and channel dimensions: (15,15) -> (1,1,15,15)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        return state_tensor.to(self.device)

    def save(self, path: str):
        """
        Save model parameters and training state.
        
        Args:
            path (str): Path to save the model
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': list(self.memory)
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """
        Load model parameters and training state.
        
        Args:
            path (str): Path to load the model from
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.memory = deque(checkpoint['memory'], maxlen=self.memory.maxlen)
            print(f"Model loaded from {path}")
        except FileNotFoundError:
            print(f"No saved model found at {path}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def eval_mode(self):
        """Set networks to evaluation mode."""
        self.policy_net.eval()
        self.target_net.eval()

    def train_mode(self):
        """Set networks to training mode."""
        self.policy_net.train()
        self.target_net.train()