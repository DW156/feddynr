import numpy as np
import argparse
import importlib
import random
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from datetime import datetime
from utils.plot_utils import plot_training_metrics, plot_algorithm_specific_metrics, save_metrics
from torchvision import datasets, transforms

# GLOBAL PARAMETERS
OPTIMIZERS = ['fedavg', 'fedprox', 'feddynr']  # 修改支持的优化器列表
DATASETS = ['sent140', 'nist', 'shakespeare', 'mnist', 
'synthetic_iid', 'synthetic_0_0', 'synthetic_0.5_0.5', 'synthetic_1_1']

MODEL_PARAMS = {
    'sent140.bag_dnn': (2,), # num_classes
    'sent140.stacked_lstm': (25, 2, 100), # seq_len, num_classes, num_hidden 
    'sent140.stacked_lstm_no_embeddings': (25, 2, 100), # seq_len, num_classes, num_hidden
    'nist.mclr': (26,),  # num_classes
    'mnist.mclr': (10,), # num_classes
    'mnist.cnn': (10,),  # num_classes
    'shakespeare.stacked_lstm': (80, 80, 256), # seq_len, emb_dim, num_hidden
    'synthetic.mclr': (10, ) # num_classes
}

def read_options():
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer',
                        help='name of optimizer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='fedavg')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        choices=DATASETS,
                        default='mnist')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='mclr')
    parser.add_argument('--num_rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=200)
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=1)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=10)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=10)
    parser.add_argument('--num_epochs', 
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=20)
    parser.add_argument('--learning_rate',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.01)
    parser.add_argument('--mu',
                        help='constant for prox;',
                        type=float,
                        default=0)  # 默认为0，即FedAvg
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=1)
    parser.add_argument('--drop_percent',
                        help='percentage of slow devices',
                        type=float,
                        default=0)
    
    # FedDynR specific parameters
    parser.add_argument('--mu_base',
                        help='base mu for FedDynR;',
                        type=float,
                        default=1.0)
    parser.add_argument('--mu_0',
                        help='initial mu for FedDynR;',
                        type=float,
                        default=1.0)
    parser.add_argument('--T',
                        help='T parameter for FedDynR;',
                        type=int,
                        default=200)
    parser.add_argument('--lambda_',
                        help='lambda parameter for FedDynR;',
                        type=float,
                        default=1.0)
    parser.add_argument('--alpha',
                        help='alpha parameter for FedDynR;',
                        type=float,
                        default=0.1)
    parser.add_argument('--num_clients',
                        help='number of clients;',
                        type=int,
                        default=100)
    parser.add_argument('--device',
                        help='device to use;',
                        type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))

    # Set seeds
    random.seed(1 + parsed['seed'])
    np.random.seed(12 + parsed['seed'])
    torch.manual_seed(123 + parsed['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123 + parsed['seed'])

    # load selected model
    if parsed['dataset'].startswith("synthetic"):  # all synthetic datasets use the same model
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'synthetic', parsed['model'])
    else:
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', parsed['dataset'], parsed['model'])

    mod = importlib.import_module(model_path)
    learner = getattr(mod, 'Model')

    # load selected trainer
    if parsed['optimizer'] in ['fedavg', 'fedprox']:
        # FedAvg 和 FedProx 使用相同的训练器，只是 mu 值不同
        opt_path = 'flearn.trainers.fedprox'
        mod = importlib.import_module(opt_path)
        optimizer = getattr(mod, 'Server')
    elif parsed['optimizer'] == 'feddynr':
        # FedDynR 使用自己的训练器
        from flearn.trainers.feddynr import FedDynRServer as optimizer

    # add selected model parameter
    parsed['model_params'] = MODEL_PARAMS['.'.join(model_path.split('.')[2:])]

    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()]);
    fmtString = '\t%' + str(maxLen) + 's : %s';
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)

    return parsed, learner, optimizer

def setup_logging(log_dir: str):
    """设置日志"""
    log_file = os.path.join(log_dir, f'training_{datetime.now():%Y%m%d_%H%M%S}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def create_clients(train_dataset, test_dataset, config):
    """创建客户端"""
    clients = []
    for i in range(config.num_clients):
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False
        )
        
        # 创建模型和客户端
        model = MCLR(config.input_dim, config.num_classes)
        client = FedDynRClient(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=config.device,
            learning_rate=config.learning_rate,
            mu_base=config.mu_base,
            mu_0=config.mu_0,
            T=config.T,
            lambda_=config.lambda_,
            alpha=config.alpha
        )
        clients.append(client)
        
        # 记录每个客户端的数据分布
        logging.info(f"Client {i} data distribution:")
        client_labels = torch.tensor([y for _, y in train_loader.dataset])
        for c in range(config.num_classes):
            count = (client_labels == c).sum().item()
            logging.info(f"  Class {c}: {count} samples")
    
    return clients

def load_mnist_data(config):
    """加载 MNIST 数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # 将数据集分配给客户端
    num_items_per_client = len(train_dataset) // config['num_clients']
    client_data_dict = {}
    
    for i in range(config['num_clients']):
        start_idx = i * num_items_per_client
        end_idx = (i + 1) * num_items_per_client if i < config['num_clients'] - 1 else len(train_dataset)
        
        client_data = {
            'x': [],
            'y': []
        }
        
        for idx in range(start_idx, end_idx):
            img, label = train_dataset[idx]
            client_data['x'].append(img.view(-1).numpy())  # 展平图像
            client_data['y'].append(label)
            
        client_data['x'] = np.array(client_data['x'])
        client_data['y'] = np.array(client_data['y'])
        client_data_dict[str(i)] = client_data
    
    # 准备测试数据
    test_data = {
        'x': [],
        'y': []
    }
    
    for img, label in test_dataset:
        test_data['x'].append(img.view(-1).numpy())
        test_data['y'].append(label)
        
    test_data['x'] = np.array(test_data['x'])
    test_data['y'] = np.array(test_data['y'])
    
    # 创建用户列表和组列表
    users = [str(i) for i in range(config['num_clients'])]
    groups = []  # 空组列表
    
    return users, groups, client_data_dict, {u: test_data for u in users}

def load_synthetic_data(config):
    """加载合成数据集"""
    # 生成合成数据
    input_dim = config['input_dim']
    num_classes = config['num_classes']
    num_samples = 1000  # 每个客户端的样本数

    client_data_dict = {}
    for i in range(config['num_clients']):
        # 生成随机特征
        x = np.random.randn(num_samples, input_dim)
        # 生成随机标签
        y = np.random.randint(0, num_classes, size=num_samples)
        
        client_data_dict[str(i)] = {
            'x': x,
            'y': y
        }
    
    # 生成测试数据
    test_x = np.random.randn(1000, input_dim)
    test_y = np.random.randint(0, num_classes, size=1000)
    test_data = {
        'x': test_x,
        'y': test_y
    }
    
    users = [str(i) for i in range(config['num_clients'])]
    groups = []
    
    return users, groups, client_data_dict, {u: test_data for u in users}

def load_shakespeare_data(config):
    """加载 Shakespeare 数据集"""
    # 这里需要实现 Shakespeare 数据集的加载逻辑
    # 由于数据集较大，可能需要从文件加载
    raise NotImplementedError("Shakespeare dataset loading not implemented yet")

def load_sent140_data(config):
    """加载 Sent140 数据集"""
    # 这里需要实现 Sent140 数据集的加载逻辑
    raise NotImplementedError("Sent140 dataset loading not implemented yet")

def load_nist_data(config):
    """加载 NIST 数据集"""
    # 这里需要实现 NIST 数据集的加载逻辑
    raise NotImplementedError("NIST dataset loading not implemented yet")

def load_data(config):
    """根据配置加载相应的数据集"""
    dataset_loaders = {
        'mnist': load_mnist_data,
        'synthetic': load_synthetic_data,
        'shakespeare': load_shakespeare_data,
        'sent140': load_sent140_data,
        'nist': load_nist_data
    }
    
    if config['dataset'] not in dataset_loaders:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")
        
    # 调用相应的数据集加载函数
    return dataset_loaders[config['dataset']](config)

def load_config(args):
    """加载并处理配置"""
    # 加载基础配置
    with open('configs/base_config.json', 'r') as f:
        config = json.load(f)
    
    # 如果指定了配置文件，用其覆盖基础配置
    if args.config:
        with open(args.config, 'r') as f:
            override_config = json.load(f)
            config.update(override_config)
    
    # 处理命令行参数
    if args.dataset:
        config['dataset'] = args.dataset
    if args.optimizer:
        config['optimizer'] = args.optimizer
    
    # 加载数据集特定配置
    if config['dataset'] in config['dataset_configs']:
        dataset_config = config['dataset_configs'][config['dataset']]
        config.update(dataset_config)
    
    # 加载优化器特定配置
    if config['optimizer'] in config['optimizer_configs']:
        optimizer_config = config['optimizer_configs'][config['optimizer']]
        config.update(optimizer_config)
    
    # 清理辅助配置
    if 'dataset_configs' in config:
        del config['dataset_configs']
    if 'optimizer_configs' in config:
        del config['optimizer_configs']
    
    return config

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Federated Learning Training')
    parser.add_argument('--config', type=str, help='Path to config file (optional)')
    parser.add_argument('--dataset', type=str, choices=DATASETS, help='Dataset to use')
    parser.add_argument('--optimizer', type=str, choices=OPTIMIZERS, help='Optimizer to use')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args)
    
    # 设置日志
    setup_logging(config['log_dir'])
    logging.info(f"Starting training with config: {config}")
    
    # 创建必要的目录
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['model_dir'], exist_ok=True)
    
    # 加载数据
    users, groups, train_data, test_data = load_data(config)
    logging.info(f"Loaded dataset with {sum(len(data['y']) for data in train_data.values())} training samples and {len(list(test_data.values())[0]['y'])} test samples")
    
    # 加载模型
    if config['dataset'].startswith("synthetic"):
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'synthetic', config['model'])
    else:
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', config['dataset'], config['model'])
    
    mod = importlib.import_module(model_path)
    learner = getattr(mod, 'Model')
    
    # 创建服务器
    if config['optimizer'] in ['fedavg', 'fedprox']:
        opt_path = 'flearn.trainers.fedprox'
        mod = importlib.import_module(opt_path)
        server = getattr(mod, 'Server')
        server = server(config, learner, (users, groups, train_data, test_data))
        server.train()
    elif config['optimizer'] == 'feddynr':
        from flearn.trainers.feddynr import FedDynRServer
        server = FedDynRServer(config, learner, (users, groups, train_data, test_data))
        server.train()
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

if __name__ == '__main__':
    main()
