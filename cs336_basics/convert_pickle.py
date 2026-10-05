import pickle
import numpy as np
import os

src = "/Volumes/T7SSD/datasets/openwebtext/valid_data.pkl"
dst = "/Volumes/T7SSD/datasets/openwebtext/valid_data.bin"

# Pass 1: count total tokens
total_tokens = 0

with open(src, "rb") as f:
    while True:
        try:
            chunk = pickle.load(f)
            total_tokens += len(chunk)
        except EOFError:
            break

print("total tokens:", total_tokens)

# Pass 2: create memmap and fill it
mm = np.memmap(
    dst,
    dtype=np.uint16,
    mode="w+",
    shape=(total_tokens,),
)

offset = 0

with open(src, "rb") as f:
    while True:
        try:
            chunk = pickle.load(f)
            chunk = np.asarray(chunk, dtype=np.uint16)

            mm[offset:offset + len(chunk)] = chunk
            offset += len(chunk)

        except EOFError:
            break

mm.flush()

print("written tokens:", offset)
print("size GB:", mm.nbytes / 1e9)