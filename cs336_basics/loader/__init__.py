import torch
import numpy as np

# probably should be a class
def data_loader(x: np.ndarray, batch_size: int, context_length: int, device: str="mps"):
    max_start = x.shape[0] - context_length # in reality it is x.size[0] - ((context_length - 1) + 1). this is because if context is 1 still same length. plus one comes from the +1 offset for the targets
    starts = np.random.randint(0, max_start, batch_size)
    offsets = np.arange(0, context_length)
    
    inputs: np.ndarray = starts[:, None] + offsets
    
    targets = inputs + 1
    
    input_ids = torch.tensor(x[inputs], dtype=torch.long, device=device)
    target_ids = torch.tensor(x[targets], dtype=torch.long, device=device)
    
    return (input_ids, target_ids)
    
    
    
    