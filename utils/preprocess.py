import numpy as np
import torch

def process_input(data, device=None):
    """
    Convert incoming JSON data into a PyTorch tensor
    Expected: data["features"] = list of 188 floats
    Returns tensor with shape (1, 1, 188) on the specified device
    
    Args:
        data: Dict with "features" key containing list of floats
        device: torch.device (defaults to CUDA if available, else CPU)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    arr = np.array(data["features"], dtype=np.float32)

    # Validate length
    if arr.ndim != 1:
        arr = arr.flatten()
    if len(arr) < 188:
        arr = np.pad(arr, (0, 188 - len(arr)), 'constant')
    elif len(arr) > 188:
        arr = arr[:188]

    tensor = torch.tensor(arr, dtype=torch.float32).view(1, 1, 188).to(device)
    return tensor
