import numpy as np
import torch
from tqdm import tqdm
from typing import List, Tuple, Dict, Any

from flearn.models.client import Client
from flearn.utils.model_utils import Metrics

class BaseFedarated(object):
    def __init__(self, params: Dict[str, Any], learner: torch.nn.Module, dataset: Tuple):
        # transfer parameters to self
        for key, val in params.items(): 
            setattr(self, key, val)

        # create worker nodes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model based on model type
        if params['model'] == 'mclr':
            # MCLR model takes input_dim and num_classes
            self.client_model = learner(params['input_dim'], params['num_classes']).to(self.device)
        elif params['model'] == 'cnn':
            # CNN model takes num_classes
            self.client_model = learner(params['num_classes']).to(self.device)
        elif params['model'] == 'stacked_lstm':
            # LSTM model takes seq_len, emb_dim, num_hidden
            self.client_model = learner(*params['model_params']).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {params['model']}")
            
        self.clients = self.setup_clients(dataset, self.client_model)
        print('{} Clients in Total'.format(len(self.clients)))
        self.latest_model = self.client_model.state_dict()

        # initialize system metrics
        self.metrics = Metrics(self.clients, params)

    def __del__(self):
        self.client_model.close()

    def setup_clients(self, dataset: Tuple, model: torch.nn.Module = None) -> List[Client]:
        '''instantiates clients based on given train and test data directories

        Return:
            list of Clients
        '''
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
        return all_clients

    def train_error_and_loss(self) -> Tuple[List[str], List[str], List[int], List[float], List[float]]:
        num_samples = []
        tot_correct = []
        losses = []

        for c in self.clients:
            ct, cl, ns = c.train_error_and_loss() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, tot_correct, losses

    def show_grads(self) -> List[np.ndarray]:  
        '''
        Return:
            gradients on all workers and the global gradient
        '''
        model_len = sum(p.numel() for p in self.client_model.parameters())
        global_grads = np.zeros(model_len)  

        intermediate_grads = []
        samples = []

        self.client_model.load_state_dict(self.latest_model)
        for c in self.clients:
            num_samples, client_grads = c.get_grads(self.latest_model) 
            samples.append(num_samples)
            global_grads = np.add(global_grads, client_grads * num_samples)
            intermediate_grads.append(client_grads)

        global_grads = global_grads * 1.0 / np.sum(np.asarray(samples)) 
        intermediate_grads.append(global_grads)

        return intermediate_grads
 
    def test(self) -> Tuple[List[str], List[str], List[int], List[float]]:
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        self.client_model.load_state_dict(self.latest_model)
        for c in self.clients:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct

    def save(self):
        pass

    def select_clients(self, round: int, num_clients: int = 20) -> Tuple[np.ndarray, List[Client]]:
        '''selects num_clients clients weighted by number of samples from possible_clients
        
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''
        num_clients = min(num_clients, len(self.clients))
        np.random.seed(round)  # make sure for each comparison, we are selecting the same clients each round
        indices = np.random.choice(range(len(self.clients)), num_clients, replace=False)
        return indices, np.asarray(self.clients)[indices]

    def aggregate(self, wsolns: List[Tuple[int, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        total_weight = 0.0
        base = {k: torch.zeros_like(v) for k, v in wsolns[0][1].items()}
        
        for (w, soln) in wsolns:  # w is the number of local samples
            total_weight += w
            for k, v in soln.items():
                base[k] += w * v

        averaged_soln = {k: v / total_weight for k, v in base.items()}
        return averaged_soln

