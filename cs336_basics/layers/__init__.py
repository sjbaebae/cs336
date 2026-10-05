from torch import nn, Tensor
import torch.functional as F
import torch
import einx


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        
        std = (2 / (in_features + out_features)) ** (1/2)
        bound = 3 * std
        weights = torch.empty((out_features, in_features), device=device, dtype=dtype)
        weights = nn.init.trunc_normal_(weights, 0, std, -1 * bound, bound)
        # W following normal Wx lin alg is (out_features, in_features)
        
        self.weights = nn.Parameter(weights)

    def forward(self, x:Tensor):
        # X shape (..., in_features)
        
        return einx.dot("... [in_feat], out_feat [in_feat] -> ... out_feat", x, self.weights)
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        weights = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        weights = nn.init.trunc_normal_(weights, 0, 1, -3, 3)
        self.weights = nn.Parameter(weights)
        
    def forward(self, x:Tensor):
        # X shape (..., k) // batch, and last dim is k where each element is a token id
        
        return einx.get_at("[num_embeddings] embedding_dim, ... k -> ... k embedding_dim", self.weights, x)

class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        gain = torch.ones((d_model), device=device, dtype=dtype)
        
        self.weights = nn.Parameter(gain)
        self.eps = torch.scalar_tensor(eps, device=device, dtype=dtype)
        
    def forward(self, x: Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        # RMS norm is sqrt(norm(x^2) + e)
        squared_mean = einx.mean("... [d_model]", x.square())
        rms_inv = torch.rsqrt(squared_mean + self.eps)
        
        return einx.multiply("... d_model, ..., d_model -> ... d_model", x, rms_inv, self.weights)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff:int, device=None, dtype=None):
        super().__init__()
        weights1 = torch.zeros((d_ff, d_model), device=device, dtype=dtype)
        weights2 = torch.zeros((d_model, d_ff), device=device, dtype=dtype)
        weights3 = torch.zeros((d_ff, d_model), device=device, dtype=dtype)
        
        std = (2 / (d_ff + d_model)) ** (1/2)
        pos_bound = 3 * std
        # init xavier
        weights1 = nn.init.trunc_normal_(weights1, 0, std, -1*pos_bound, pos_bound)
        weights2 = nn.init.trunc_normal_(weights2, 0, std, -1*pos_bound, pos_bound)
        weights3 = nn.init.trunc_normal_(weights3, 0, std, -1*pos_bound, pos_bound)
        
        self.weights1 = nn.Parameter(weights1)
        self.weights2 = nn.Parameter(weights2)
        self.weights3 = nn.Parameter(weights3)
    
    def forward(self, x:Tensor):
        w1_x = einx.dot("... [d_model], d_ff [d_model] -> ... d_ff", x, self.weights1)
        gate = einx.multiply("... d_ff, ... d_ff -> ... d_ff", w1_x, torch.sigmoid(w1_x))
        
        w3_x = einx.dot("... [d_model], d_ff [d_model] -> ... d_ff", x, self.weights3)
        silued_x = einx.multiply("... d_ff, ... d_ff", gate, w3_x)
        
        return einx.dot("... [d_ff], d_model [d_ff] -> ... d_model", silued_x, self.weights2)
        
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: str):
        super().__init__()
        
        indices = torch.arange(max_seq_len, device=device).unsqueeze(-1).expand(-1, 1)
        ks = torch.arange(d_k // 2, device=device).expand(1, -1)
        
        exponent = (2 * ks) / d_k
        angles = indices / torch.pow(theta, exponent) # (seq_len, d_k // 2)
        
        s = torch.sin(angles)
        c = torch.cos(angles)
        
        R = torch.stack([c, -s, s, c], dim=-1)
        R = einx.id("... (a b) -> ... a b", R, a=2, b=2)
        
        # now for each we have (seq_len, d_k // 2, 2, 2)
        
        self.register_buffer("rotation", R, persistent=False)
        
    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        R_selected = self.rotation[token_positions]
        
        x = einx.id("... (half_d_model a) -> ... half_d_model a", x, a=2)
        x = einx.dot("... s dk b, s dk a b -> ... s dk a", x, R_selected)
        
        return einx.id("... dk a -> ... (dk a)", x)
        
def sdpa(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor):
    d_k = Q.size(-1)
    
    logits = einx.dot("b ... s_q [d_k], b ... s_k [d_k] -> b ... s_q s_k", Q, K) /(d_k ** (1/2))
    
    logits = logits.masked_fill(~mask, float("-inf"))
    
    attn = softmax(logits, -1)
    
    return einx.dot("... s_q [s_k], ... [s_k] v -> ... s_q v", attn, V)
    
    
    
    
def softmax(x: Tensor, dim: int, temp=1):
    maxs = x.max(dim=dim, keepdim=True).values
    exponentials = torch.exp((x - maxs) / temp) # with temp scaling
    
    return exponentials / exponentials.sum(dim=dim, keepdim=True)

class MHSA(nn.Module):
    def __init__(self, d_model:int, num_heads: int, rope: RoPE = None, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_model_head = d_model // num_heads
        
        std = (2 / (d_model + d_model)) ** (1/2)
        bound = 3 * std
        weights = torch.zeros((3, d_model, d_model), device=device, dtype=dtype)
        weights = nn.init.trunc_normal_(weights, 0, std, -1 * bound, bound)
        
        weights_out = torch.zeros((d_model, d_model), device=device, dtype=dtype) #(d_model, d_model1)
        weights_out = nn.init.trunc_normal_(weights_out, 0, std, -1 * bound, bound)
        
        self.weights = nn.Parameter(weights)
        self.weights_out = nn.Parameter(weights_out)
        self.rope = rope
        
    def forward(self, x: Tensor, token_positions: Tensor = None):
        
        x = einx.dot("... [d_model], k d_model1 [d_model] -> k ... d_model1", x, self.weights) # in form k, ..., d_model1
        x = einx.id("k batch ... (h d_model_head) -> k batch h ... d_model_head", x, h=self.num_heads) # k (3), batch, heads, seq_len, ..., d_model_head
        
        Qs = x[0]
        Ks = x[1]
        Vs = x[2]
        
        q_seq_len = Qs.size(-2)
        k_seq_len = Ks.size(-2)
        
        if self.rope:
            assert token_positions is not None, "Token positions must not be none if using RoPE"
            Qs = self.rope.forward(Qs, token_positions) #batch, heads, seq_len d_model_head
            Ks = self.rope.forward(Ks, token_positions) #batch, heads, seq_len d_model_head
            
        # now apply attention on each head
        mask = torch.tril(torch.ones(q_seq_len, k_seq_len, device=x.device)).bool()
        x = sdpa(Qs, Ks, Vs, mask) #batch heads, seq_len, d_model_head
        x = einx.id("b h ... d_model_head -> b ... (h d_model_head)", x) #batch, seq_len, d_model1
        
        # final out
        return einx.dot("... [d_model1], d_model [d_model1] -> ... d_model", x, self.weights_out)
        
        
        