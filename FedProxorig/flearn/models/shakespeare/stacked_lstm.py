import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from typing import Dict, Tuple, List, Optional

import os
import sys
import tensorflow as tf

from tensorflow.contrib import rnn

utils_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
utils_dir = os.path.join(utils_dir, 'utils')
sys.path.append(utils_dir)

from model_utils import batch_data, batch_data_multiple_iters
from language_utils import letter_to_vec, word_to_indices
from tf_utils import graph_size
from tf_utils import process_sparse_grad

def process_x(raw_x_batch: List[str]) -> np.ndarray:
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    x_batch = np.array(x_batch)
    return x_batch

def process_y(raw_y_batch: List[str]) -> List[np.ndarray]:
    y_batch = [letter_to_vec(c) for c in raw_y_batch]
    return y_batch

class StackedLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, num_layers: int = 2):
        super(StackedLSTM, self).__init__()
        self.embedding = nn.Embedding(input_size, 8)
        self.lstm = nn.LSTM(8, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, 8)
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_size)
        # Take the last output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        output = self.fc(last_output)  # (batch_size, num_classes)
        return output

class Model(object):
    def __init__(self, seq_len: int, num_classes: int, n_hidden: int, optimizer: Optional[optim.Optimizer] = None, seed: int = 1):
        # Set random seed
        torch.manual_seed(123 + seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(123 + seed)
            
        # Create model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        
        # Create LSTM model
        self.model = StackedLSTM(num_classes, n_hidden, num_classes).to(self.device)
        
        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer
            
        # Calculate model size and FLOPs
        self.size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        self.flops = sum(p.numel() for p in self.model.parameters())  # Approximate FLOPs
    
    def set_params(self, model_params: Optional[Dict[str, torch.Tensor]] = None):
        if model_params is not None:
            self.model.load_state_dict(model_params)

    def get_params(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()

    def get_gradients(self, data: Dict[str, np.ndarray], model_len: int) -> Tuple[int, np.ndarray]:
        '''in order to avoid the OOM error, we need to calculate the gradients on each 
        client batch by batch. batch size here is set to be 100.

        Return: a one-D array (after flattening all gradients)
        '''
        self.model.train()
        grads = np.zeros(model_len)
        num_samples = len(data['y'])
        processed_samples = 0

        if num_samples < 50:
            input_data = process_x(data['x'])
            target_data = process_y(data['y'])
            
            # Convert to tensors
            x = torch.LongTensor(input_data).to(self.device)
            y = torch.FloatTensor(target_data).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = nn.functional.cross_entropy(output, y)
            loss.backward()
            
            # Get gradients
            grad_idx = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    grad = param.grad.cpu().numpy().flatten()
                    grads[grad_idx:grad_idx + len(grad)] = grad
                    grad_idx += len(grad)
                    
            processed_samples = num_samples
        else:
            # Calculate gradients in batches of 50
            for i in range(min(int(num_samples / 50), 4)):
                input_data = process_x(data['x'][50*i:50*(i+1)])
                target_data = process_y(data['y'][50*i:50*(i+1)])
                
                # Convert to tensors
                x = torch.LongTensor(input_data).to(self.device)
                y = torch.FloatTensor(target_data).to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = nn.functional.cross_entropy(output, y)
                loss.backward()
                
                # Get gradients
                grad_idx = 0
                for param in self.model.parameters():
                    if param.grad is not None:
                        grad = param.grad.cpu().numpy().flatten()
                        grads[grad_idx:grad_idx + len(grad)] += grad
                        grad_idx += len(grad)
                        
            grads = grads * 1.0 / min(int(num_samples/50), 4)
            processed_samples = min(int(num_samples / 50), 4) * 50

        return processed_samples, grads
    
    def solve_inner(self, data: Dict[str, np.ndarray], num_epochs: int = 1, batch_size: int = 32) -> Tuple[Dict[str, torch.Tensor], int]:
        '''Solves local optimization problem'''
        self.model.train()
        
        for _ in trange(num_epochs, desc='Epoch: ', leave=False):
            for X, y in batch_data(data, batch_size):
                input_data = process_x(X)
                target_data = process_y(y)
                
                # Convert to tensors
                x = torch.LongTensor(input_data).to(self.device)
                y = torch.FloatTensor(target_data).to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = nn.functional.cross_entropy(output, y)
                loss.backward()
                self.optimizer.step()
                
        soln = self.get_params()
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops
        return soln, comp

    def solve_iters(self, data: Dict[str, np.ndarray], num_iters: int = 1, batch_size: int = 32) -> Tuple[Dict[str, torch.Tensor], int]:
        '''Solves local optimization problem'''
        self.model.train()
        
        for X, y in batch_data_multiple_iters(data, batch_size, num_iters):
            input_data = process_x(X)
            target_data = process_y(y)
            
            # Convert to tensors
            x = torch.LongTensor(input_data).to(self.device)
            y = torch.FloatTensor(target_data).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = nn.functional.cross_entropy(output, y)
            loss.backward()
            self.optimizer.step()
            
        soln = self.get_params()
        comp = 0  # We don't track FLOPs for this method
        return soln, comp
    
    def test(self, data: Dict[str, np.ndarray]) -> Tuple[int, float]:
        '''Tests the model on the given data'''
        self.model.eval()
        x_vecs = process_x(data['x'])
        labels = process_y(data['y'])
        
        # Convert to tensors
        x = torch.LongTensor(x_vecs).to(self.device)
        y = torch.FloatTensor(labels).to(self.device)
        
        with torch.no_grad():
            output = self.model(x)
            loss = nn.functional.cross_entropy(output, y)
            pred = output.argmax(dim=1)
            true_labels = y.argmax(dim=1)
            tot_correct = (pred == true_labels).sum().item()
            
        return tot_correct, loss.item()
    
    def close(self):
        """Clean up resources"""
        pass  # No need to close anything in PyTorch

