import torch
import copy
import numpy as np
from typing import List, Dict, Any, Optional
from ..models.feddynr_model import MCLR
import json
import os
import logging
from torch.utils.data import DataLoader

class FedDynRServer:
    """FedDynR服务器实现"""
    
    def __init__(self,
                 global_model: MCLR,
                 num_clients: int,
                 train_loader: DataLoader,
                 log_dir: str = "logs",
                 mu_base: float = 1.0,
                 mu_0: float = 1.0,
                 T: int = 100,
                 lambda_: float = 0.1,
                 alpha: float = 0.5):
        """
        初始化FedDynR服务器
        
        Args:
            global_model: 全局模型
            num_clients: 客户端数量
            train_loader: 全局训练数据加载器
            log_dir: 日志目录
            mu_base: 基础正则化系数
            mu_0: 初始μ值
            T: 衰减周期
            lambda_: 方向一致性约束系数
            alpha: 历史锚点衰减系数
        """
        self.global_model = global_model
        self.num_clients = num_clients
        self.train_loader = train_loader
        self.log_dir = log_dir
        self.mu_base = mu_base
        self.mu_0 = mu_0
        self.T = T
        self.lambda_ = lambda_
        self.alpha = alpha
        
        # 初始化历史最优模型
        self.historical_model = copy.deepcopy(global_model)
        self.best_accuracy = 0.0
        self.round_number = 0
        
        # 计算全局数据分布
        self.global_dist = self._compute_global_distribution()
        
        # 初始化指标记录
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = {
            'train_loss': [],
            'test_accuracy': [],
            'grad_diff': [],
            'heterogeneity': [],
            'mu_values': [],
            'alpha_values': []
        }
    
    def _compute_global_distribution(self) -> torch.Tensor:
        """计算全局数据的类别分布"""
        class_counts = torch.zeros(self.global_model.linear.out_features)
        for _, target in self.train_loader:
            for i in range(self.global_model.linear.out_features):
                class_counts[i] += (target == i).sum().item()
        return class_counts / class_counts.sum()
        
    def get_global_model(self) -> MCLR:
        """获取全局模型"""
        return self.global_model
    
    def get_historical_model(self) -> MCLR:
        """获取历史最优模型"""
        return self.historical_model
    
    def get_global_distribution(self) -> torch.Tensor:
        """获取全局数据分布"""
        return self.global_dist
    
    def aggregate(self,
                 client_results: List[Dict[str, Any]],
                 client_weights: List[float],
                 round_number: int) -> None:
        """
        聚合客户端模型
        
        Args:
            client_results: 客户端训练结果列表
            client_weights: 客户端权重列表
            round_number: 当前训练轮次
        """
        self.round_number = round_number
        
        # 归一化权重
        total_weight = sum(client_weights)
        client_weights = [w/total_weight for w in client_weights]
        
        # 初始化新的全局模型
        new_global_state = {}
        
        # 提取客户端模型并聚合
        client_models = [result['model_state'] for result in client_results]
        for key in self.global_model.state_dict().keys():
            new_global_state[key] = torch.zeros_like(self.global_model.state_dict()[key])
            for i, client_model in enumerate(client_models):
                new_global_state[key] += client_weights[i] * client_model[key]
        
        # 更新全局模型
        self.global_model.load_state_dict(new_global_state)
        
        # 更新历史最优模型
        avg_accuracy = float(np.mean([result.get('accuracy', 0) for result in client_results]))
        if avg_accuracy > self.best_accuracy:
            self.best_accuracy = avg_accuracy
            self.historical_model.load_state_dict(new_global_state)
        
        # 记录指标
        self._log_metrics(client_results)
        
    def _log_metrics(self, client_results: List[Dict[str, Any]]):
        """
        记录训练指标
        
        Args:
            client_results: 客户端训练结果列表
        """
        # 计算平均指标并转换为Python原生float类型
        avg_train_loss = float(np.mean([r['train_loss'] for r in client_results]))
        avg_grad_diff = float(np.mean([r.get('grad_diff', 0) for r in client_results]))
        avg_heterogeneity = float(np.mean([r['heterogeneity_score'] for r in client_results]))
        avg_mu = float(np.mean([r['mu'] for r in client_results]))
        avg_alpha = float(np.mean([r['alpha'] for r in client_results]))
        
        # 存储指标
        self.metrics['train_loss'].append(avg_train_loss)
        self.metrics['test_accuracy'].append(self.best_accuracy)
        self.metrics['grad_diff'].append(avg_grad_diff)
        self.metrics['heterogeneity'].append(avg_heterogeneity)
        self.metrics['mu_values'].append(avg_mu)
        self.metrics['alpha_values'].append(avg_alpha)
        
        # 记录到日志
        logging.info(f"Round {self.round_number + 1}:")
        logging.info(f"  Train Loss: {avg_train_loss:.4f}")
        logging.info(f"  Test Accuracy: {self.best_accuracy:.2f}%")
        logging.info(f"  Gradient Difference: {avg_grad_diff:.4f}")
        logging.info(f"  Heterogeneity Score: {avg_heterogeneity:.4f}")
        logging.info(f"  Dynamic Mu: {avg_mu:.4f}")
        logging.info(f"  Alpha: {avg_alpha:.4f}")
        
        # 保存到JSON文件
        metrics_file = os.path.join(self.log_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    def save_model(self, path: str) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'global_model': self.global_model.state_dict(),
            'historical_model': self.historical_model.state_dict(),
            'best_accuracy': self.best_accuracy,
            'round_number': self.round_number,
            'global_dist': self.global_dist
        }, path)
    
    def load_model(self, path: str) -> None:
        """
        加载模型
        
        Args:
            path: 加载路径
        """
        checkpoint = torch.load(path)
        self.global_model.load_state_dict(checkpoint['global_model'])
        self.historical_model.load_state_dict(checkpoint['historical_model'])
        self.best_accuracy = checkpoint['best_accuracy']
        self.round_number = checkpoint['round_number']
        self.global_dist = checkpoint.get('global_dist', self._compute_global_distribution()) 