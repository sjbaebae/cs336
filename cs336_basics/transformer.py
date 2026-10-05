import einx
import torch
from torch import nn, Tensor

from .layers import Linear, RMSNorm, MHSA, RoPE, SwiGLU, Embedding, softmax

class PreNormTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.rope = RoPE(theta, d_model // num_heads, max_seq_len, device=device)
        self.rms1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rms2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.mhsa = MHSA(d_model, num_heads, self.rope, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
    
    def forward(self, x:Tensor, token_positions: Tensor = None):
        if token_positions is None:
            shape = x.shape
            seq_len = shape[1]
            token_positions = torch.arange(0, seq_len, device=x.device) #seq_len
        x2 = self.rms1(x)
        x2 = self.mhsa(x2, token_positions)
        x2 = x2 + x
        
        # FFN next
        x3 = self.rms2(x2)
        x3 = self.ffn(x3)
        x3 = x3 + x2
        
        return x3
        
    
class Transformer_LM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers, d_model: int, num_heads: int, d_ff: int, theta: float, device=None, dtype=None) -> None:
        super().__init__()
        
        self.embedding = Embedding(vocab_size, d_model, device, dtype)
        self.blocks = nn.ModuleList([PreNormTransformerBlock(d_model, num_heads, d_ff, theta, context_length, device, dtype) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.unembedding = Linear(d_model, vocab_size, device, dtype)
        
    def forward(self, ids: Tensor):
        # embed ids
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.unembedding(x)
        
        return logits