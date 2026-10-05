import torch
from typing import Iterable

def grad_clipping(params: Iterable[torch.nn.Parameter], max_l2, eps=1e-6):
    
    params = list(params)
    
    squared_sum = 0
    
    for p in params:
        if p.grad is None:
            continue
        
        squared_sum += (p.grad.data.square().sum())
        
    L2_norm = torch.sqrt(squared_sum)
    
    if L2_norm > max_l2:
        scaling_factor = (max_l2) / (L2_norm + eps)
        
        for p in params:
            if p.grad is None:
                continue
            
            p.grad.mul_(scaling_factor)