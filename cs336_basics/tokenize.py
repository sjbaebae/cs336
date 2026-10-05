from .tokenizer import Tokenizer
from argparse import ArgumentParser
import os
from pathlib import Path
import pickle
from rust_module import RustTokenizer
from pathlib import Path
from tqdm import tqdm
import numpy as np

tinystories_dataset_path = os.path.join(Path(__file__).parent.resolve(), "../datasets/tinystories")
owt_dataset_path = "/Volumes/T7SSD/datasets/openwebtext" # T7
tinystories_output_path = os.path.join(Path(__file__).parent.resolve(), "../outputs/tinystories")
owt_output_path = "/Users/sbae703/Research/ML/cs336/assignment1-basics/outputs/openwebtext"
def tokenize_tinystories():
    special_tokens = ["<|endoftext|>"]
    tiny_tokenizer = RustTokenizer.from_file(os.path.join(tinystories_output_path, "vocab.pkl"), os.path.join(tinystories_output_path, "merges.pkl"), special_tokens, None)
    
    # encode train data
    chunk_size = 1024 * 1024 * 2 #in chars
    dataset_train_path = os.path.join(tinystories_dataset_path, "TinyStoriesV2-GPT4-train.txt")
    with open(dataset_train_path) as f, tqdm(total=os.path.getsize(dataset_train_path), unit="B", unit_scale=True) as bar:
        with open(os.path.join(tinystories_dataset_path, "train_data.bin"), "wb") as out:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                if not chunk.endswith("\n"):
                    chunk += f.readline()
                    
                encoded_chunk = tiny_tokenizer.encode(chunk)
                np.asarray(encoded_chunk, dtype=np.uint16).tofile(out)
                bar.update(f.tell() - bar.n)
                
    dataset_valid_path = os.path.join(tinystories_dataset_path, "TinyStoriesV2-GPT4-valid.txt")
    with open(dataset_valid_path) as f, tqdm(total=os.path.getsize(dataset_valid_path), unit="B", unit_scale=True) as bar:
        with open(os.path.join(tinystories_dataset_path, "valid_data.bin"), "wb") as out:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                    
                if not chunk.endswith("\n"):
                    chunk += f.readline()
                
                encoded_chunk = tiny_tokenizer.encode(chunk)
                np.asarray(encoded_chunk, dtype=np.uint16).tofile(out)
                bar.update(f.tell() - bar.n)
        
    
def tokenize_owt():
    special_tokens = ["<|endoftext|>"]
    owt_tokenizer = RustTokenizer.from_file(os.path.join(owt_output_path, "vocab.pkl"), os.path.join(owt_output_path, "merges.pkl"), special_tokens, None)
    
        
    # encode train data
    chunk_size = 1024 * 1024 * 2 #in chars
    dataset_train_path = os.path.join(owt_dataset_path, "owt_train.txt")
    with open(dataset_train_path) as f, tqdm(total=os.path.getsize(dataset_train_path), unit="B", unit_scale=True) as bar:
        with open(os.path.join(owt_dataset_path, "train_data.pkl"), "wb") as out:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                if not chunk.endswith("\n"):
                    chunk += f.readline()
                    
                encoded_chunk = owt_tokenizer.encode(chunk)
                pickle.dump(encoded_chunk, out)
                bar.update(f.tell() - bar.n)
                
    dataset_valid_path = os.path.join(owt_dataset_path, "owt_valid.txt")
    with open(dataset_valid_path) as f, tqdm(total=os.path.getsize(dataset_valid_path), unit="B", unit_scale=True) as bar:
        with open(os.path.join(owt_dataset_path, "valid_data.pkl"), "wb") as out:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                    
                if not chunk.endswith("\n"):
                    chunk += f.readline()
                
                encoded_chunk = owt_tokenizer.encode(chunk)
                pickle.dump(encoded_chunk, out)
                bar.update(f.tell() - bar.n)
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dataset")
    
    args = parser.parse_args()
    
    if args.dataset == "tinystories":
        tokenize_tinystories()
    if args.dataset == "owt":
        tokenize_owt()