import torch
from torch import Tensor
from .tokenizer import Tokenizer
from .layers import softmax

def sample(logits: Tensor, temp=0.8, top_p = 0.9):
    probs = softmax(logits, -1, temp=temp)
    sorted_probs, sorted_ids = probs.sort(descending=True)
    
    # find cumsum > top_p
    cumsum = sorted_probs.cumsum(dim=-1)
    remove = cumsum > top_p
    remove[..., 1:] = remove[..., :-1].clone() #shift by one tok that crosses top_p stays
    remove[..., 0] = 0
    
    sorted_probs = sorted_probs.masked_fill(remove, 0) 
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True) # renormalize since removed a bunch
    choice = torch.multinomial(sorted_probs, num_samples=1)
    next_id = sorted_ids.gather(-1, choice)
    
    return next_id
    

def decode(query: str, tokenizer: Tokenizer, model: torch.nn.Module, max_len=4096, temp=0.8, top_p=0.9):
    encoded_ids = torch.tensor([tokenizer.encode(query)], dtype=torch.long, device=next(model.parameters()).device)
    end_of_word_id = next(i for i, b in tokenizer.vocab.items() if b == b"<|endoftext|>")
    
    logits = model(encoded_ids)[:, -1, :] # last token
    next_id = sample(logits, temp, top_p)
    
    result = "" + tokenizer.decode([next_id.item()])
    generated_toks = 1
    encoded_ids = torch.cat([encoded_ids, next_id], dim=-1)
    
    # print(result)
    
    while next_id != end_of_word_id and generated_toks < max_len:
        logits = model(encoded_ids)[:, -1, :]
        next_id = sample(logits, temp, top_p)
        
        result += tokenizer.decode([next_id.item()])
        encoded_ids = torch.cat([encoded_ids, next_id], dim=-1)
        
        generated_toks += 1
        # print(result)
        
    return result
        
        
        
        
    
        