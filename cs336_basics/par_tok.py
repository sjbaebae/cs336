import multiprocessing as mp
from typing import BinaryIO


def find_chunk_boundaries(file: BinaryIO, n_processes: int, split_special_token: bytes) -> list[int]:
    return []


def pretok_par(filepath: str, n_processes: int = 4, split_special_token: bytes = b"<|endoftext|>"):
    # import to pathlib Path
    with open(filepath, "rb") as f:
        boundaries: list[int] = find_chunk_boundaries(f, n_processes, split_special_token)

        # take chunk boundaries (multirocessor step)
        # first get each pair for the chunks zip(boundaries[:-1], boundaries[1:]) -> now [0,b], [b,c]... and do a multiprocessor run
        #
        #
