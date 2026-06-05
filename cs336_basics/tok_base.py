from slow_tok import pretokloop


## Usage
def pretok(filepath: str, n_processes: int = 4, split_special_token: bytes = b"<|endoftext|>"):
    # import to pathlib Path
    pretokloop(filepath, n_processes, split_special_token)
    #
