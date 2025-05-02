import matplotlib.pyplot as plt
import numpy as np
import json
import os
from typing import Dict, List, Optional, Union

def plot_training_metrics(
    metrics: Dict[str, List[float]], 
    save_path: str,
    algorithm: str = "fedavg",
    show_heterogeneity: bool = True,
    show_grad_diff: bool = True
):
    """
    绘制训练指标
    
    Args:
        metrics: 训练指标字典
        save_path: 保存路径
        algorithm: 算法名称，可选 "fedavg", "fedprox", "feddynr"
        show_heterogeneity: 是否显示异质性分数
        show_grad_diff: 是否显示梯度差异
    """
    # 根据算法确定子图数量
    n_plots = 2
    if show_heterogeneity:
        n_plots += 1
    if show_grad_diff:
        n_plots += 1
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # 绘制训练损失
    axes[0].plot(metrics['train_loss'])
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Loss')
    
    # 绘制测试准确率
    axes[1].plot(metrics['test_accuracy'])
    axes[1].set_title('Test Accuracy')
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Accuracy (%)')
    
    # 根据算法和参数绘制其他指标
    plot_idx = 2
    
    if show_grad_diff and 'grad_diff' in metrics:
        axes[plot_idx].plot(metrics['grad_diff'])
        axes[plot_idx].set_title('Gradient Difference')
        axes[plot_idx].set_xlabel('Round')
        axes[plot_idx].set_ylabel('Difference')
        plot_idx += 1
    
    if show_heterogeneity and 'heterogeneity' in metrics:
        axes[plot_idx].plot(metrics['heterogeneity'])
        axes[plot_idx].set_title('Heterogeneity Score')
        axes[plot_idx].set_xlabel('Round')
        axes[plot_idx].set_ylabel('Score')
        plot_idx += 1
    
    # 隐藏多余的子图
    for i in range(plot_idx, 4):
        axes[i].set_visible(False)
    
    # 添加算法名称到标题
    plt.suptitle(f'{algorithm.upper()} Training Metrics', fontsize=16)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(save_path)
    plt.close()

def plot_algorithm_specific_metrics(
    metrics: Dict[str, List[float]], 
    save_path: str,
    algorithm: str = "fedavg"
):
    """
    绘制算法特定的指标
    
    Args:
        metrics: 训练指标字典
        save_path: 保存路径
        algorithm: 算法名称，可选 "fedavg", "fedprox", "feddynr"
    """
    if algorithm == "fedprox" and 'mu_values' in metrics:
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # 绘制μ值
        ax.plot(metrics['mu_values'])
        ax.set_title('FedProx μ Values')
        ax.set_xlabel('Round')
        ax.set_ylabel('μ')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图形
        plt.savefig(save_path)
        plt.close()
    
    elif algorithm == "feddynr" and all(key in metrics for key in ['mu_values', 'alpha_values']):
        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 绘制动态μ值
        axes[0].plot(metrics['mu_values'])
        axes[0].set_title('Dynamic μ Values')
        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('μ')
        
        # 绘制α值
        axes[1].plot(metrics['alpha_values'])
        axes[1].set_title('α Values')
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('α')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图形
        plt.savefig(save_path)
        plt.close()

def save_metrics(metrics: Dict[str, List[float]], save_path: str):
    """
    保存训练指标
    
    Args:
        metrics: 训练指标字典
        save_path: 保存路径
    """
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def load_metrics(load_path: str) -> Dict[str, List[float]]:
    """
    加载训练指标
    
    Args:
        load_path: 加载路径
        
    Returns:
        Dict[str, List[float]]: 训练指标字典
    """
    with open(load_path, 'r') as f:
        return json.load(f)

def plot_comparison(
    metrics_list: List[Dict[str, List[float]]],
    algorithm_names: List[str],
    save_path: str,
    metric_name: str = "test_accuracy"
):
    """
    比较不同算法的性能
    
    Args:
        metrics_list: 不同算法的指标列表
        algorithm_names: 算法名称列表
        save_path: 保存路径
        metric_name: 要比较的指标名称
    """
    plt.figure(figsize=(10, 6))
    
    for metrics, name in zip(metrics_list, algorithm_names):
        if metric_name in metrics:
            plt.plot(metrics[metric_name], label=name)
    
    plt.title(f'Comparison of {metric_name.replace("_", " ").title()}')
    plt.xlabel('Round')
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() 