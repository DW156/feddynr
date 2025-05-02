import pickle
from typing import List, TypeVar, Any, Dict
import matplotlib.pyplot as plt
from utils.plot_utils import save_metrics, load_metrics, plot_training_metrics

T = TypeVar('T')

def save_obj(obj: Any, name: str) -> None:
    """Save an object to a pickle file.
    
    Args:
        obj: The object to save
        name: The base name of the file (without extension)
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name: str) -> Any:
    """Load an object from a pickle file.
    
    Args:
        name: The base name of the file (without extension)
        
    Returns:
        The loaded object
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def iid_divide(l: List[T], g: int) -> List[List[T]]:
    '''Divide list l among g groups.
    Each group has either int(len(l)/g) or int(len(l)/g)+1 elements.
    
    Args:
        l: The list to divide
        g: Number of groups
        
    Returns:
        A list of groups
    '''
    num_elems = len(l)
    group_size = int(len(l)/g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    
    # Create small groups
    for i in range(num_small_groups):
        glist.append(l[group_size*i:group_size*(i+1)])
        
    # Create big groups
    bi = group_size*num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi+group_size*i:bi+group_size*(i+1)])
        
    return glist

def plot_training_metrics(metrics: Dict[str, List[float]], save_path: str, algorithm: str = "fedavg"):
    """
    根据算法类型自动绘制训练指标
    """
    # 决定要画哪些指标
    plot_items = [("train_loss", "Training Loss", "Loss"),
                  ("train_accuracy", "Training Accuracy", "Accuracy (%)"),
                  ("test_accuracy", "Test Accuracy", "Accuracy (%)")]
    if algorithm in ["fedprox", "feddynr"] and "grad_diff" in metrics:
        plot_items.append(("grad_diff", "Gradient Difference", "Difference"))
    if algorithm == "feddynr":
        if "mu_values" in metrics:
            plot_items.append(("mu_values", "Dynamic μ Values", "μ"))
        if "alpha_values" in metrics:
            plot_items.append(("alpha_values", "α Values", "α"))
        if "heterogeneity" in metrics:
            plot_items.append(("heterogeneity", "Heterogeneity Score", "Score"))

    n_plots = len(plot_items)
    nrows = (n_plots + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(15, 5 * nrows))
    axes = axes.flatten()

    for idx, (key, title, ylabel) in enumerate(plot_items):
        if key in metrics:
            axes[idx].plot(metrics[key])
            axes[idx].set_title(title)
            axes[idx].set_xlabel('Round')
            axes[idx].set_ylabel(ylabel)
        else:
            axes[idx].set_visible(False)

    # 隐藏多余子图
    for i in range(len(plot_items), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'{algorithm.upper()} Training Metrics', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

metrics = {
    "train_loss": [...],
    "train_accuracy": [...],
    "test_accuracy": [...],
    "grad_diff": [...],
    "mu_values": [...],
    "heterogeneity": [...],
    "alpha_values": [...],
}

plot_training_metrics(metrics, "FedProxorig/plots/xxx.png", algorithm="fedavg")
plot_training_metrics(metrics, "FedProxorig/plots/xxx.png", algorithm="fedprox")
plot_training_metrics(metrics, "FedProxorig/plots/xxx.png", algorithm="feddynr")

# 训练主循环后
from utils.plot_utils import save_metrics

metrics_path = f"FedProxorig/metrics/{algorithm}_metrics.json"
save_metrics(metrics, metrics_path)

for algorithm in ["fedavg", "fedprox", "feddynr"]:
    metrics_path = f"FedProxorig/metrics/{algorithm}_metrics.json"
    plot_path = f"FedProxorig/plots/{algorithm}_training_metrics.png"
    metrics = load_metrics(metrics_path)
    plot_training_metrics(metrics, plot_path, algorithm=algorithm)