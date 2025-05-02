import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Dict, Any, Optional
from ..models.feddynr_model import (
    compute_heterogeneity_score,
    compute_dynamic_mu,
    compute_direction_loss,
    compute_anchor_loss
)

class FedDynRClient:
    """FedDynR客户端实现"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 test_loader: Optional[torch.utils.data.DataLoader] = None,
                 device: str = "cpu",
                 learning_rate: float = 0.01,
                 mu_base: float = 1.0,
                 mu_0: float = 1.0,
                 T: int = 200,
                 lambda_: float = 1.0,
                 alpha: float = 0.1):
        """
        初始化FedDynR客户端
        
        Args:
            model: 本地模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            device: 计算设备
            learning_rate: 学习率
            mu_base: 基础正则化系数
            mu_0: 初始μ值
            T: 衰减周期
            lambda_: 方向一致性约束系数
            alpha: 历史锚点衰减系数
        """
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        
        # FedDynR parameters
        self.mu_base = mu_base
        self.mu_0 = mu_0
        self.T = T
        self.lambda_ = lambda_
        self.alpha = alpha
        
        # Initialize optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize dynamic parameters
        self.h = torch.zeros_like(self.flatten_params()).to(device)
        
    def flatten_params(self):
        """将模型参数展平为一维向量"""
        return torch.cat([p.data.view(-1) for p in self.model.parameters()])
        
    def get_model_params(self):
        """获取模型参数"""
        return self.flatten_params()
        
    def train(self,
              global_model: nn.Module,
              historical_model: Optional[nn.Module] = None,
              round_number: int = 0,
              global_dist: Optional[torch.Tensor] = None,
              **kwargs) -> Dict[str, Any]:
        """
        训练本地模型
        
        Args:
            global_model: 全局模型
            historical_model: 历史最优模型
            round_number: 当前训练轮次
            global_dist: 全局数据分布
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 训练结果
        """
        self.model.train()
        
        # 复制全局模型参数
        self.global_model = type(global_model)(
            global_model.linear.in_features,
            global_model.linear.out_features
        ).to(self.device)
        self.global_model.load_state_dict(global_model.state_dict())
        
        # 复制历史模型参数
        if historical_model is not None:
            self.historical_model = type(historical_model)(
                historical_model.linear.in_features,
                historical_model.linear.out_features
            ).to(self.device)
            self.historical_model.load_state_dict(historical_model.state_dict())
        else:
            self.historical_model = None
        
        # 计算本地数据分布
        local_dist = self._compute_class_distribution()
        
        # 使用服务器提供的全局分布或计算均匀分布
        if global_dist is not None:
            global_dist = global_dist.to(self.device)
        else:
            global_dist = torch.ones(self.model.linear.out_features, device=self.device) / self.model.linear.out_features
        
        # 计算异质性分数
        heterogeneity_score = compute_heterogeneity_score(local_dist, global_dist)
        
        # 计算动态mu
        mu = compute_dynamic_mu(
            mu_0=self.mu_0,
            t=round_number,
            T=self.T,
            heterogeneity_score=heterogeneity_score
        )
        
        # 计算alpha
        alpha = self.alpha ** (round_number / self.T)
        
        # 训练循环
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 计算本地梯度
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.local_grad = {name: param.grad.clone() for name, param in self.model.named_parameters()}
            
            # 计算全局梯度
            global_output = self.global_model(data)
            global_loss = self.criterion(global_output, target)
            self.global_model.zero_grad()
            global_loss.backward()
            self.global_grad = {name: param.grad.clone() for name, param in self.global_model.named_parameters()}
            
            # 计算方向感知正则化损失
            dir_loss = compute_direction_loss(self.local_grad, self.global_grad)
            
            # 计算锚点损失
            if self.historical_model:
                anchor_loss = compute_anchor_loss(
                    dict(self.model.named_parameters()),
                    dict(self.global_model.named_parameters()),
                    dict(self.historical_model.named_parameters()),
                    alpha
                )
            else:
                anchor_loss = compute_anchor_loss(
                    dict(self.model.named_parameters()),
                    dict(self.global_model.named_parameters()),
                    None,
                    alpha
                )
            
            # 总损失
            total_loss = loss + mu * anchor_loss + self.lambda_ * dir_loss
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        # 计算平均训练损失
        train_loss /= num_batches
        
        # 计算梯度差异
        grad_diff = 0.0
        for local_param, global_param in zip(self.model.parameters(), self.global_model.parameters()):
            grad_diff += torch.norm(local_param.data - global_param.data).item()
        
        # 更新 h
        with torch.no_grad():
            self.h = self.h - self.lambda_ * (self.flatten_params() - global_dist)
        
        return {
            'model_state': self.model.state_dict(),
            'train_loss': train_loss,
            'grad_diff': grad_diff,
            'heterogeneity_score': heterogeneity_score,
            'mu': mu,
            'alpha': alpha
        }
    
    def test(self) -> Dict[str, float]:
        """
        测试本地模型
        
        Returns:
            Dict[str, float]: 测试结果
        """
        if not self.test_loader:
            return {}
            
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return {
            'test_loss': test_loss / len(self.test_loader),
            'accuracy': 100. * correct / total
        }
    
    def _compute_class_distribution(self) -> torch.Tensor:
        """计算本地数据类别分布"""
        class_counts = torch.zeros(self.model.linear.out_features, device=self.device)
        for _, target in self.train_loader:
            target = target.to(self.device)
            for i in range(self.model.linear.out_features):
                class_counts[i] += (target == i).sum().item()
        return class_counts / class_counts.sum() 