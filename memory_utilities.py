import torch
import gc
from typing import Tuple, List, Optional, Dict, Any, Union

def get_gpu_memory_info() -> Tuple[float, float]:
    """
    Returns current GPU memory usage and total memory in GB.
    
    Returns:
        Tuple[float, float]: (used_memory_gb, total_memory_gb)
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0
    
    device = torch.cuda.current_device()
    total_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    reserved_memory = torch.cuda.memory_reserved(device)
    allocated_memory = torch.cuda.memory_allocated(device)
    used_memory_gb = allocated_memory / (1024**3)
    
    return used_memory_gb, total_memory_gb

def clear_gpu_memory() -> None:
    """
    Clears GPU cache and runs garbage collection to free up memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def auto_device_placement(data_size: int, 
                          feature_dim: int, 
                          min_free_memory_gb: float = 2.0) -> str:
    """
    Determines whether to use GPU or CPU based on data size and available memory.
    
    Args:
        data_size: Number of data points
        feature_dim: Dimensionality of features
        min_free_memory_gb: Minimum free GPU memory required (GB)
        
    Returns:
        str: 'cuda' if GPU should be used, 'cpu' otherwise
    """
    if not torch.cuda.is_available():
        return 'cpu'
    
    # Estimate memory requirements for pairwise distance calculation
    # This is a very rough estimate and could be improved
    estimated_memory_gb = (data_size**2 * 4 * 3) / (1024**3) # Assume 3x overhead
    
    used_memory_gb, total_memory_gb = get_gpu_memory_info()
    free_memory_gb = total_memory_gb - used_memory_gb
    
    if free_memory_gb > estimated_memory_gb + min_free_memory_gb:
        return 'cuda'
    else:
        print(f"Warning: Insufficient GPU memory. Estimated need: {estimated_memory_gb:.2f} GB, Free: {free_memory_gb:.2f} GB")
        return 'cpu'

def optimize_memory_usage(X: torch.Tensor, 
                         Y: torch.Tensor, 
                         param: torch.Tensor,
                         target_device: str = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Attempts to optimize memory usage by moving tensors to appropriate device.
    
    Args:
        X: Feature tensor
        Y: Target tensor
        param: Parameter tensor (alpha or theta)
        target_device: Target device ('cuda' or 'cpu'), autodetected if None
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Optimized tensors on appropriate device
    """
    if target_device is None:
        target_device = auto_device_placement(X.shape[0], X.shape[1])
    
    # If we're moving to CPU, try to free GPU memory first
    if target_device == 'cpu' and any(t.is_cuda for t in [X, Y, param]):
        clear_gpu_memory()
    
    # Move tensors to target device
    X_device = X.to(target_device)
    Y_device = Y.to(target_device)
    param_device = param.to(target_device)
    
    return X_device, Y_device, param_device