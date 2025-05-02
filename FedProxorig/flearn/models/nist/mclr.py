import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from typing import Dict, Tuple, List, Optional

from flearn.utils.model_utils import batch_data, batch_data_multiple_iters

class MCLR(nn.Module):
    """Multi-class Logistic Regression model for NIST"""
    def __init__(self, input_dim: int, num_classes: int):
        super(MCLR, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class Model(object):
    '''
    Assumes that images are 28px by 28px
    '''
    
    def __init__(self, num_classes: int, optimizer: Optional[optim.Optimizer] = None, seed: int = 1):
        # Set random seed
        torch.manual_seed(123 + seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(123 + seed)
            
        # Create model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MCLR(784, num_classes).to(self.device)
        self.num_classes = num_classes
        
        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
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
        self.model.train()
        grads = np.zeros(model_len)
        num_samples = len(data['y'])
        
        # Convert data to tensors
        x = torch.FloatTensor(data['x']).to(self.device)
        y = torch.LongTensor(data['y']).to(self.device)
        
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
                
        return num_samples, grads
    
    def solve_inner(self, data: Dict[str, np.ndarray], num_epochs: int = 1, batch_size: int = 32) -> Tuple[Dict[str, torch.Tensor], int]:
        '''Solves local optimization problem'''
        self.model.train()
        
        for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120):
            for X, y in batch_data(data, batch_size):
                X = X.to(self.device)
                y = y.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(X)
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
            X = X.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = nn.functional.cross_entropy(output, y)
            loss.backward()
            self.optimizer.step()
            
        soln = self.get_params()
        comp = 0  # We don't track FLOPs for this method
        return soln, comp
    
    def test(self, data: Dict[str, np.ndarray]) -> Tuple[int, float]:
        '''Tests the model on the given data'''
        self.model.eval()
        x = torch.FloatTensor(data['x']).to(self.device)
        y = torch.LongTensor(data['y']).to(self.device)
        
        with torch.no_grad():
            output = self.model(x)
            loss = nn.functional.cross_entropy(output, y)
            pred = output.argmax(dim=1)
            tot_correct = (pred == y).sum().item()
            
        return tot_correct, loss.item()
    
    def close(self):
        """Clean up resources"""
        pass  # No need to close anything in PyTorch
