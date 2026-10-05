from .bpe_trainer import BPETrainer
import os
import cProfile
import json
import pathlib
import pickle

from argparse import ArgumentParser

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], num_chunks=64, profile=False, pretoken_count_file: pathlib.Path | str = None, save=False, pretoken_count_save_file: pathlib.Path | str = None) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    BPE = BPETrainer(input_path, vocab_size, special_tokens)
    print("starting training...")
    return BPE.train(num_chunks, profile, pretoken_count_file, save, pretoken_count_save_file)

def train_bpe_tinystories(profile=False):
    main_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
    input_path = os.path.join(main_dir, "datasets/tinystories/TinyStoriesV2-GPT4-train.txt")
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens, profile=profile)
    
    output_dir = os.path.join(main_dir, "outputs/tinystories")
    
    # build file if they do not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(os.path.join(output_dir, "vocab.pkl"), "bw") as pf:
        pickle.dump(vocab, pf)
        
    with open(os.path.join(output_dir, "merges.pkl"), "bw") as pf:
        pickle.dump(merges, pf)
        

def train_bpe_owt(profile=False):
    main_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
    input_path = os.path.join(main_dir, "/Volumes/T7SSD/datasets/openwebtext/owt_train.txt")
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]
    
    output_dir = os.path.join(main_dir, "outputs/openwebtext")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens, num_chunks=1024, profile=profile, pretoken_count_file=os.path.join(output_dir, "pretoken_counts.pkl"))
    
    with open(os.path.join(output_dir, "vocab.pkl"), "bw") as pf:
        pickle.dump(vocab, pf)
        
    with open(os.path.join(output_dir, "merges.pkl"), "bw") as pf:
        pickle.dump(merges, pf)
        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("-p", "--profile", action="store_true")
    
    args = parser.parse_args()
    
    if args.dataset == "tinystories":
        if args.profile:
            cProfile.run(f'train_bpe_tinystories()', "profiling_results")
        else:
            train_bpe_tinystories()
    if args.dataset == "openwebtext":
        train_bpe_owt()