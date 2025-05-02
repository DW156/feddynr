import torch
import torch.optim as optim
from typing import Dict, Optional

class PerturbedGradientDescent(optim.Optimizer):
    """Implementation of Perturbed Gradient Descent, i.e., FedProx optimizer"""
    def __init__(self, params, lr=0.001, mu=0.01):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= mu:
            raise ValueError(f"Invalid mu value: {mu}")
            
        defaults = dict(lr=lr, mu=mu)
        super(PerturbedGradientDescent, self).__init__(params, defaults)
        
        # Initialize vstar for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['vstar'] = torch.zeros_like(p)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                state = self.state[p]
                vstar = state['vstar']
                
                # Update vstar
                vstar.add_(group['mu'] * (p.data - vstar))
                
                # Update parameter
                p.data.add_(-group['lr'] * (p.grad.data + group['mu'] * (p.data - vstar)))

        return loss

    def set_params(self, global_params: Dict[str, torch.Tensor], model: torch.nn.Module):
        """Set the global parameters as vstar for each parameter.

        Args:
            global_params: Dictionary of global parameters
            model: The model whose parameters will be updated
        """
        for name, param in model.named_parameters():
            if name in global_params:
                state = self.state[param]
                state['vstar'].copy_(global_params[name])
