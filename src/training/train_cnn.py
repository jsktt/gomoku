# src/training/train_cnn.py

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from ..utils.config import Config
import numpy as np

class GomokuDataset(Dataset):
    def __init__(self, states, policies, values):
        # Process states to create two channels: one for black pieces and one for white
        processed_states = []
        for state in states:
            black_channel = (state == 1).astype(np.float32)
            white_channel = (state == -1).astype(np.float32)
            processed_state = np.stack([black_channel, white_channel])
            processed_states.append(processed_state)
        
        self.states = torch.FloatTensor(processed_states)  # Shape: [N, 2, 15, 15]
        self.policies = torch.FloatTensor(policies)
        self.values = torch.FloatTensor(values)
        
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]

def train_cnn(model, train_data, num_epochs=100, batch_size=32):
    """Train the CNN model."""
    states, policies, values = train_data
    dataset = GomokuDataset(states, policies, values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()  # Set to training mode
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_states, batch_policies, batch_values in progress_bar:
            # Move batch data to device
            batch_states = batch_states.to(model.device)  # Shape: [batch_size, 2, 15, 15]
            batch_policies = batch_policies.to(model.device)
            batch_values = batch_values.to(model.device)
            
            loss = model.train_network(batch_states, batch_policies, batch_values)
            total_loss += loss
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss:.4f}'})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            model.save(f"{Config.CHECKPOINT_DIR}/cnn_model_epoch_{epoch+1}.pt")
    
    # Save final model
    model.save(f"{Config.CHECKPOINT_DIR}/cnn_final.pt")