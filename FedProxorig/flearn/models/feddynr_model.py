import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class MCLR(nn.Module):
    """多类逻辑回归模型"""
    
    def __init__(self, input_dim: int, num_classes: int):
        """
        初始化多类逻辑回归模型
        
        Args:
            input_dim: 输入维度
            num_classes: 类别数量
        """
        super(MCLR, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 输出张量
        """
        return self.linear(x)

def compute_heterogeneity_score(local_dist: torch.Tensor, global_dist: torch.Tensor) -> float:
    """
    计算异质性分数
    
    Args:
        local_dist: 本地数据分布
        global_dist: 全局数据分布
        
    Returns:
        float: 异质性分数
    """
    return float(torch.norm(local_dist - global_dist).item())

def compute_dynamic_mu(mu_0: float, t: int, T: int, heterogeneity_score: float) -> float:
    """
    计算动态μ值
    
    Args:
        mu_0: 初始μ值
        t: 当前轮次
        T: 衰减周期
        heterogeneity_score: 异质性分数
        
    Returns:
        float: 动态μ值
    """
    return mu_0 * (1 - t / T) * (1 + heterogeneity_score)

def compute_direction_loss(local_grad: Dict[str, torch.Tensor], 
                         global_grad: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    计算方向一致性损失
    
    Args:
        local_grad: 本地梯度
        global_grad: 全局梯度
        
    Returns:
        torch.Tensor: 方向一致性损失
    """
    loss = 0.0
    for name in local_grad.keys():
        local_grad_norm = torch.norm(local_grad[name])
        global_grad_norm = torch.norm(global_grad[name])
        if local_grad_norm > 0 and global_grad_norm > 0:
            cos_sim = F.cosine_similarity(local_grad[name].view(-1), global_grad[name].view(-1), dim=0)
            loss += (1 - cos_sim) / 2
    return loss

def compute_anchor_loss(local_params: Dict[str, torch.Tensor],
                       global_params: Dict[str, torch.Tensor],
                       historical_params: Optional[Dict[str, torch.Tensor]],
                       alpha: float) -> torch.Tensor:
    """
    计算锚点损失
    
    Args:
        local_params: 本地模型参数
        global_params: 全局模型参数
        historical_params: 历史模型参数
        alpha: 历史锚点衰减系数
        
    Returns:
        torch.Tensor: 锚点损失
    """
    loss = 0.0
    for name in local_params.keys():
        if historical_params is not None:
            loss += torch.norm(local_params[name] - (1 - alpha) * global_params[name] - alpha * historical_params[name])
        else:
            loss += torch.norm(local_params[name] - global_params[name])
    return loss 