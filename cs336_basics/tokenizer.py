import json
import pickle
from typing import Iterable
import regex
from mpire import WorkerPool
import rust_module


GPT_PAT = r"""'(?:[sdmt]|ve|re|ll)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT_REGEX = regex.compile(GPT_PAT)

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.special_tokens = special_tokens
        self.vocab = vocab
        if self.special_tokens:
            sorted_special_toks = sorted(self.special_tokens, key=len, reverse=True)
            self.split_pat = "(" + "|".join([regex.escape(r) for r in sorted_special_toks]) + ")"
        self.merges = merges
        
        reverse_vocab = {
            v:k for (k,v) in vocab.items()
        }
        merges_ints_to_tok = []
        for (left_bytes, right_bytes) in merges:
            joined_bytes = left_bytes + right_bytes
            new_tok_id = reverse_vocab[joined_bytes]
            left_int = reverse_vocab[left_bytes]
            right_int = reverse_vocab[right_bytes]
            
            merges_ints_to_tok.append((left_int, right_int, new_tok_id))
            
        self.rust_tokenizer = rust_module.RustTokenizer(reverse_vocab, merges_ints_to_tok)
    
    @classmethod
    def from_file(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
            
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
            
        return Tokenizer(vocab, merges, special_tokens)
    
    def _pretokenize_text(self, text, regex_PAT=GPT_REGEX):
        if self.special_tokens:
            subtexts = regex.split(self.split_pat, text)
            subtexts_without_splits = subtexts[::2]
            special_tok_chunks = subtexts[1::2]
        else:
            subtexts_without_splits = [text]
            special_tok_chunks = []

        pretokenized_subtexts = [
            [m.group().encode("utf-8") for m in regex_PAT.finditer(subtext)]
            for subtext in subtexts_without_splits
        ]

        return pretokenized_subtexts, [
            s.encode("utf-8") for s in special_tok_chunks
        ]
            
    def encode(self, text:str) -> list[int]: # Encode a text input into a sequence of IDs
        # first pretokenize
        import time
        t0 = time.perf_counter()
        pretokens_chunks, special_tok_chunks = self._pretokenize_text(text) # each chunk split by special token
        print("Time taken pretokenization", time.perf_counter() - t0)
        t1 = time.perf_counter()
        results = self.rust_tokenizer.prepare_and_run_merges(pretokens_chunks, special_tok_chunks)
        print("Time taken merging: ", time.perf_counter() - t1)
        return results
    
    def encode_from_file(self, input_path:str, output_path: str):
        pass
            
            
    def encode_iterable(self, iterable: Iterable[str]):
        for data in iterable:
            yield from self.encode(data)
            
    
    def decode(self, ids: list[int]) -> str:
        final_bytes = b"".join(self.vocab[id] for id in ids)
        return final_bytes.decode("utf-8", errors='ignore')