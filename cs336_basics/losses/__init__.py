from torch import nn, Tensor
import torch.functional as F
import torch
import einx

def cross_entropy(logits: Tensor, targets: Tensor):
    # logits (batch, seq_len, vocab)
    max_logits = logits.max(-1, keepdim=True).values
    
    # targets (batch, seq_len - 1)
    
    safe_sum = torch.exp(logits - max_logits).sum(-1)
    logsumexp = torch.log(safe_sum) # (batch, seq_len)
    
    # return is CE per token so (batch, seq_len)
    target_logits = einx.get_at("... [vocab], ... -> ...", logits, targets)
    losses = -target_logits + max_logits.squeeze(-1) + logsumexp
    
    return losses.mean()
    
    