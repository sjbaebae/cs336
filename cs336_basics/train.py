from cs336_basics.losses import cross_entropy
from .checkpointing import load_checkpoint, save_checkpoint
from .transformer import Transformer_LM
from .optimizers import AdamW
from .loader import data_loader
import numpy as np
import typer
import wandb
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer()

def load_data(data_file: str):
    return np.memmap(data_file)

def build_model():
    return Transformer_LM()
   
@app.command()
def main(
    data_path: str = "/Users/sbae703/Research/ML/cs336/assignment1-basics/datasets/tinystories/train_data.bin",
    vocab_size: int = 50512,
    d_model: int = 512,
    d_ff: int = 1365, # 8/3
    num_heads: int = 8,
    context_length: int = 256,
    num_layers: int = 4,
    batch_size: int = 8,
    iterations: int = 50,
    theta: float = 10000.0,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    load_ckpt: str | None = None,
    save_ckpt: str | None = None,
    device="mps",
    team_entity: str ="sbae12-university-of-rochester",
    project: str ="cs336-runs"
):
    assert save_ckpt is not None, "save checkpoint filepath must not be None"
    print("loading model...")
    model = Transformer_LM(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, theta, device)
    optim = AdamW(model.parameters(), lr, weight_decay, (beta1, beta2), eps)
    start = load_checkpoint(load_ckpt, model, optim) if load_ckpt else 0
    
    print('loading file...')
    mm = np.memmap(data_path, dtype=np.uint16)
    
    print("loaded file")
    print("starting training run")
    
    # When this block exits, it waits for logged data to finish uploading.
    # If an exception is raised, the run is marked failed.
    with wandb.init(entity=team_entity, project=project) as run:
        # Save model inputs and hyperparameters.
        run.config.learning_rate = lr

        # Run your experiment code.
        for step in range(iterations):
            # Do some training...
            optim.zero_grad()
            (data_batch, target_batch) = data_loader(mm, batch_size, context_length, device)
            logits = model(data_batch)
            loss = cross_entropy(logits, target_batch)
            loss.backward()
            optim.step()

                # Log metrics over time to visualize model performance.
            run.log({"loss": loss})

        # Upload model outputs as artifacts.
        # run.log_artifact(model)
        
    # save model
    save_checkpoint(model, optim, start + iterations, save_ckpt)        
    

if __name__ == "__main__":
    app()
    