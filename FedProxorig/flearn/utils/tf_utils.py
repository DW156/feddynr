import numpy as np
import torch
from typing import List, Union, Tuple

def __num_elems(shape: Tuple[int, ...]) -> int:
    '''Returns the number of elements in the given shape

    Args:
        shape: shape tuple
    
    Return:
        tot_elems: int
    '''
    tot_elems = 1
    for s in shape:
        tot_elems *= int(s)
    return tot_elems

def model_size(model: torch.nn.Module) -> int:
    '''Returns the size of the given model in bytes

    The size of the model is calculated by summing up the sizes of each
    parameter. The sizes of parameters are calculated by multiplying
    the number of bytes in their dtype with their number of elements.

    Args:
        model: PyTorch model
    Return:
        integer representing size of model (in bytes)
    '''
    tot_size = 0
    for param in model.parameters():
        tot_elems = __num_elems(param.shape)
        dtype_size = param.element_size()
        var_size = tot_elems * dtype_size
        tot_size += var_size
    return tot_size

def process_sparse_grad(grads: List[torch.Tensor]) -> np.ndarray:
    '''
    Args:
        grads: grad returned by LSTM model (only for the shakespaere dataset)
    Return:
        a flattened grad in numpy (1-D array)
    '''
    # Convert sparse tensor to dense
    first_layer_dense = torch.zeros((80, 8))
    if grads[0].is_sparse:
        indices = grads[0]._indices()
        values = grads[0]._values()
        for i in range(indices.shape[1]):
            first_layer_dense[indices[0, i], :] = values[i, :]
    else:
        first_layer_dense = grads[0]

    client_grads = first_layer_dense.cpu().numpy().flatten()
    for i in range(1, len(grads)):
        client_grads = np.append(client_grads, grads[i].cpu().numpy().flatten())

    return client_grads

def process_grad(grads: List[torch.Tensor]) -> np.ndarray:
    '''
    Args:
        grads: grad 
    Return:
        a flattened grad in numpy (1-D array)
    '''
    client_grads = grads[0].cpu().numpy().flatten()

    for i in range(1, len(grads)):
        client_grads = np.append(client_grads, grads[i].cpu().numpy().flatten())

    return client_grads

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    '''Returns the cosine similarity between two arrays a and b
    
    Args:
        a: first array
        b: second array
        
    Returns:
        float: cosine similarity
    '''  
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product * 1.0 / (norm_a * norm_b)




