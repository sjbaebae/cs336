from typing import Callable, Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError("learning rate must be greater than 0")
        
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= (lr / (math.sqrt(t + 1))) * grad #prevent div by 0
                state["t"] = t + 1
                
        return loss
        
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError("learning rate must be greater than 0")
        
        defaults = {"lr": lr, "beta1": betas[0], "beta2": betas[1], "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            base_lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay_lambda = group["weight_decay"]
            eps = group["eps"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                m = state.get("m", 0) # first moment
                v = state.get("v", 0) # second moment
                t = state.get("t", 1) # iteration
                
                
                # apply weight decay
                p.data -= base_lr * weight_decay_lambda * p.data
                
                # get adjusted lr for gradients
                lr = base_lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t + eps)
                
                # get gradient
                grad = p.grad.data
                
                # update first moment
                state["m"] = beta1 * m + (1 - beta1) * grad
                # update second moment
                state["v"] = beta2 * v + (1 - beta2) * torch.square(grad)
                
                # update parameters
                p.data -= lr * (state["m"] / (torch.sqrt(state["v"]) + eps)) # 1e-8 to ensure numerical stability
                
                state["t"] = t + 1
            
            return loss
                
                
                
                
            
            
            
            