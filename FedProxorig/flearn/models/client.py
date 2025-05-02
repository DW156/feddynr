import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Any

class ClientDataset(Dataset):
    def __init__(self, data: Dict[str, np.ndarray]):
        self.x = torch.FloatTensor(data['x'])
        self.y = torch.LongTensor(data['y'])
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class Client(object):
    def __init__(self, id: int, group: str = None, 
                 train_data: Dict[str, np.ndarray] = {'x':[],'y':[]}, 
                 eval_data: Dict[str, np.ndarray] = {'x':[],'y':[]}, 
                 model: torch.nn.Module = None):
        self.model = model
        self.id = id  # integer
        self.group = group
        self.train_data = ClientDataset(train_data)
        self.eval_data = ClientDataset(eval_data)
        self.num_samples = len(self.train_data)
        self.test_samples = len(self.eval_data)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def load_state_dict(self, model_params: Dict[str, torch.Tensor]):
        '''set model parameters'''
        self.model.load_state_dict(model_params)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        '''get model parameters'''
        return self.model.state_dict()

    def get_grads(self, model_len: int) -> Tuple[int, np.ndarray]:
        '''get model gradient'''
        self.model.train()
        train_loader = DataLoader(self.train_data, batch_size=32, shuffle=True)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            output = self.model(x)
            loss = torch.nn.functional.cross_entropy(output, y)
            loss.backward()
            
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.cpu().numpy().flatten())
        
        return self.num_samples, np.concatenate(grads)

    def solve_inner(self, num_epochs: int = 1, batch_size: int = 10) -> Tuple[Tuple[int, Dict[str, torch.Tensor]], Tuple[int, int, int]]:
        '''Solves local optimization problem
        
        Return:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes read: number of bytes received
            2: comp: number of FLOPs executed in training process
            2: bytes_write: number of bytes transmitted
        '''
        self.model.train()
        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        
        for epoch in range(num_epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = torch.nn.functional.cross_entropy(output, y)
                loss.backward()
                optimizer.step()
        
        bytes_w = sum(p.numel() * p.element_size() for p in self.model.parameters())
        soln = self.model.state_dict()
        comp = sum(p.numel() for p in self.model.parameters()) * num_epochs * self.num_samples
        bytes_r = bytes_w
        
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)

    def solve_iters(self, num_iters: int = 1, batch_size: int = 10) -> Tuple[Tuple[int, Dict[str, torch.Tensor]], Tuple[int, int, int]]:
        '''Solves local optimization problem

        Return:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes read: number of bytes received
            2: comp: number of FLOPs executed in training process
            2: bytes_write: number of bytes transmitted
        '''
        self.model.train()
        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        
        for _ in range(num_iters):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = torch.nn.functional.cross_entropy(output, y)
                loss.backward()
                optimizer.step()
        
        bytes_w = sum(p.numel() * p.element_size() for p in self.model.parameters())
        soln = self.model.state_dict()
        comp = sum(p.numel() for p in self.model.parameters()) * num_iters * self.num_samples
        bytes_r = bytes_w
        
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)

    def train_error_and_loss(self) -> Tuple[int, float, int]:
        self.model.eval()
        train_loader = DataLoader(self.train_data, batch_size=32, shuffle=False)
        tot_correct = 0
        tot_loss = 0
        
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = torch.nn.functional.cross_entropy(output, y)
                pred = output.argmax(dim=1)
                tot_correct += (pred == y).sum().item()
                tot_loss += loss.item() * len(y)
        
        return tot_correct, tot_loss / self.num_samples, self.num_samples

    def test(self) -> Tuple[int, int]:
        '''tests current model on local eval_data

        Return:
            tot_correct: total #correct predictions
            test_samples: int
        '''
        self.model.eval()
        test_loader = DataLoader(self.eval_data, batch_size=32, shuffle=False)
        tot_correct = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                pred = output.argmax(dim=1)
                tot_correct += (pred == y).sum().item()
        
        return tot_correct, self.test_samples
