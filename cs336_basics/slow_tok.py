import os
from typing import BinaryIO


def find_chunk_boundaries(file: BinaryIO, num_chunks: int, split_special_token: bytes):
    """
    Chunk the file parts that can be pretokenized independently.
    Attempts to maintain evenly distributed chunk size but avoids breaking on non split_special_token
    Returns fewer than chunk parts if boundaries are overlapping
    """
    # find size and start point of file
    file.seek(0, os.SEEK_END)
    file_size: int = file.tell()
    # reset to initial position
    file.seek(0)

    # preallocated an array for the max num of chunks. with evenly distribution
    chunk_size: int = file_size // num_chunks
    chunk_boundaries: list[int] = [i * chunk_size for i in range(num_chunks + 1)]
    # set last chunk to be EOF
    chunk_boundaries[-1] = file_size

    # set minichunk size (read in size) -> 4kb
    minichunk_size: int = 4096

    # Skip start (0 bound). next bound -> keep minichunking until we get to special token or EOF
    for bi in range(1, num_chunks):
        initial_pos: int = chunk_boundaries[bi]

        while True:
            minichunk: bytes = file.read(minichunk_size)

            # see if we are at EOF
            if minichunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find special token in chunk
            found_at = minichunk.find(split_special_token)

            if found_at != -1:
                chunk_boundaries[bi] = initial_pos + found_at
                break
            initial_pos += found_at

    return sorted(set(chunk_boundaries))  # removes redundant, sorted


## Usage
def pretokloop(filepath: str, n_processes: int = 4, split_special_token: bytes = b"<|endoftext|>"):
    # import to pathlib Path
    with open(filepath, "rb") as f:
        boundaries: list[int] = find_chunk_boundaries(f, n_processes, split_special_token)

        # take chunk boundaries (multirocessor step)
        # first get each pair for the chunks zip(boundaries[:-1], boundaries[1:]) -> now [0,b], [b,c]... and do a multiprocessor run
        #
        #
