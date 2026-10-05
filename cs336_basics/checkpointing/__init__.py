import torch
import os
import typing


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    model_state_dict = model.state_dict()
    optim_state_dict = optimizer.state_dict()
    
    model_dict = {
        "model": model_state_dict,
        "optim": optim_state_dict,
        "iteration": iteration
    }
    
    torch.save(model_dict, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    model_dict = torch.load(src)
    
    model_state_dict = model_dict["model"]
    optim_state_dict = model_dict["optim"]
    iteration = model_dict["iteration"]
    
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optim_state_dict)
    
    return iteration