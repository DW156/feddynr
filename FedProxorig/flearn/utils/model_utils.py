import json
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Generator, Any
import random
from torchvision import datasets, transforms

class FederatedDataset(Dataset):
    def __init__(self, data: Dict[str, np.ndarray]):
        self.x = torch.FloatTensor(data['x'])
        self.y = torch.LongTensor(data['y'])
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def batch_data(data: Dict[str, np.ndarray], batch_size: int) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both torch.Tensor of length: batch_size
    '''
    dataset = FederatedDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for x, y in dataloader:
        yield x, y

def batch_data_multiple_iters(data: Dict[str, np.ndarray], batch_size: int, num_iters: int) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    dataset = FederatedDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for _ in range(num_iters):
        for x, y in dataloader:
            yield x, y

def read_data(dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join('data', dataset, 'data', 'train')
        file_path = os.path.join(train_data_dir, f'{idx}.json')
    else:
        test_data_dir = os.path.join('data', dataset, 'data', 'test')
        file_path = os.path.join(test_data_dir, f'{idx}.json')

    with open(file_path, 'r') as inf:
        cdata = json.load(inf)
    return cdata

def create_non_iid_mnist_distribution(train_dataset, num_clients, num_classes=10, min_classes_per_client=2, max_classes_per_client=3):
    """创建非IID的MNIST数据分布"""
    # 按类别分组数据
    class_indices = [[] for _ in range(num_classes)]
    for idx, label in enumerate(train_dataset.targets):
        class_indices[label.item()].append(idx)
    
    # 随机打乱每个类别的索引
    for indices in class_indices:
        random.shuffle(indices)
    
    # 为每个客户端分配类别
    client_class_distribution = []
    for i in range(num_clients):
        num_classes_for_client = random.randint(min_classes_per_client, max_classes_per_client)
        available_classes = list(range(num_classes))
        random.shuffle(available_classes)
        client_classes = available_classes[:num_classes_for_client]
        client_class_distribution.append(client_classes)
    
    # 为每个客户端分配数据
    client_indices = [[] for _ in range(num_clients)]
    class_pointers = [0] * num_classes
    
    # 确保每个客户端至少获得一定数量的样本
    min_samples_per_client = 100
    
    # 第一轮分配：确保每个客户端从其分配的类别中获得最小数量的样本
    for client_id, classes in enumerate(client_class_distribution):
        samples_needed = min_samples_per_client
        while samples_needed > 0:
            for class_id in classes:
                if class_pointers[class_id] < len(class_indices[class_id]):
                    client_indices[client_id].append(class_indices[class_id][class_pointers[class_id]])
                    class_pointers[class_id] += 1
                    samples_needed -= 1
                if samples_needed <= 0:
                    break
    
    # 第二轮分配：分配剩余的样本
    remaining_indices = []
    for class_id in range(num_classes):
        remaining_indices.extend(class_indices[class_id][class_pointers[class_id]:])
    random.shuffle(remaining_indices)
    
    # 将剩余的样本均匀分配给客户端
    for idx, index in enumerate(remaining_indices):
        client_id = idx % num_clients
        client_indices[client_id].append(index)
    
    return client_indices

def load_mnist_data(config):
    # 加载MNIST数据集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    # 获取数据集大小和客户端数量
    num_clients = config['num_clients']
    
    # 创建非IID分布
    client_indices = create_non_iid_mnist_distribution(train_dataset, num_clients)
    
    # 分配数据给每个客户端
    users = [f'user_{i}' for i in range(num_clients)]
    groups = [f'group_{i}' for i in range(num_clients)]
    train_data = {}
    test_data = {}

    for i in range(num_clients):
        # 提取客户端训练数据
        indices = client_indices[i]
        client_train_data = {
            'x': train_dataset.data[indices].numpy(),
            'y': train_dataset.targets[indices].numpy()
        }
        train_data[users[i]] = client_train_data

        # 测试数据分配（这里简单起见，每个客户端使用相同的测试集）
        test_data[users[i]] = {
            'x': test_dataset.data.numpy(),
            'y': test_dataset.targets.numpy()
        }

    return users, groups, train_data, test_data

class Metrics(object):
    def __init__(self, clients: List[Any], params: Dict[str, Any]):
        self.params = params
        num_rounds = params['num_rounds']
        self.bytes_written = {c.id: [0] * num_rounds for c in clients}
        self.client_computations = {c.id: [0] * num_rounds for c in clients}
        self.bytes_read = {c.id: [0] * num_rounds for c in clients}      
        self.accuracies = []
        self.train_accuracies = []

    def update(self, rnd: int, cid: str, stats: Tuple[int, int, int]):
        bytes_w, comp, bytes_r = stats
        self.bytes_written[cid][rnd] += bytes_w
        self.client_computations[cid][rnd] += comp
        self.bytes_read[cid][rnd] += bytes_r

    def write(self):
        metrics = {}
        metrics['dataset'] = self.params['dataset']
        metrics['num_rounds'] = self.params['num_rounds']
        metrics['eval_every'] = self.params['eval_every']
        metrics['learning_rate'] = self.params['learning_rate']
        metrics['mu'] = self.params['mu']
        metrics['num_epochs'] = self.params['num_epochs']
        metrics['batch_size'] = self.params['batch_size']
        metrics['accuracies'] = self.accuracies
        metrics['train_accuracies'] = self.train_accuracies
        metrics['client_computations'] = self.client_computations
        metrics['bytes_written'] = self.bytes_written
        metrics['bytes_read'] = self.bytes_read
        
        metrics_dir = os.path.join('out', self.params['dataset'], 
                                 f"metrics_{self.params['seed']}_{self.params['optimizer']}_{self.params['learning_rate']}_{self.params['num_epochs']}_{self.params['mu']}.json")
        
        if not os.path.exists(os.path.join('out', self.params['dataset'])):
            os.makedirs(os.path.join('out', self.params['dataset']))
            
        with open(metrics_dir, 'w') as ouf:
            json.dump(metrics, ouf)
