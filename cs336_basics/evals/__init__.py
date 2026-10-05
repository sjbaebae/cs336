from torch import Tensor
import torch

def perplexity(ce_losses: Tensor):
   return torch.exp(ce_losses.mean())