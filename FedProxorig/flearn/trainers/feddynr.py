import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm

class FedDynRServer:
    def __init__(self, global_model, num_clients, train_loader, log_dir,
                 mu_base=1.0, mu_0=1.0, T=200, lambda_=1.0, alpha=0.1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model = global_model.to(self.device)
        self.num_clients = num_clients
        self.train_loader = train_loader
        self.log_dir = log_dir
        
        # FedDynR parameters
        self.mu_base = mu_base
        self.mu_0 = mu_0
        self.T = T
        self.lambda_ = lambda_
        self.alpha = alpha
        
        # Initialize dynamic parameters
        self.h = torch.zeros_like(self.flatten_params(self.global_model)).to(self.device)
        self.current_round = 0
        
        # Metrics
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        
    def flatten_params(self, model):
        """将模型参数展平为一维向量"""
        return torch.cat([p.data.view(-1) for p in model.parameters()])
        
    def get_global_distribution(self):
        """获取全局分布"""
        return self.flatten_params(self.global_model)
        
    def aggregate(self, clients):
        """聚合客户端模型"""
        # 计算当前轮次的 mu
        t = self.current_round + 1
        mu_t = self.mu_0 * (1 - t/self.T) + self.mu_base * (t/self.T)
        
        # 收集所有客户端的模型参数
        client_params = []
        for client in clients:
            params = client.get_model_params()
            client_params.append(params)
            
        # 计算平均模型参数
        avg_params = torch.mean(torch.stack(client_params), dim=0)
        
        # 更新 h
        self.h = self.h - self.lambda_ * (avg_params - self.flatten_params(self.global_model))
        
        # 更新全局模型
        new_params = avg_params + mu_t * self.h
        self.update_global_model(new_params)
        
        self.current_round += 1
        
    def update_global_model(self, new_params):
        """使用新参数更新全局模型"""
        start = 0
        for param in self.global_model.parameters():
            num_params = param.numel()
            param.data = new_params[start:start+num_params].view(param.size())
            start += num_params
            
    def evaluate(self, test_dataset):
        """评估全局模型"""
        self.global_model.eval()
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
        
        total_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                total_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
        accuracy = correct / total
        avg_loss = total_loss / len(test_loader)
        
        self.test_losses.append(avg_loss)
        self.test_accuracies.append(accuracy)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        } 