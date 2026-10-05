# will need mmap (file read)
# multiprocessing (mpire)
import rust_module
import heapq
import pathlib
import pickle
import profile
import pstats
from typing import BinaryIO
import regex
from mpire import WorkerPool
import mmap
import os
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import itertools
from tqdm import tqdm
from line_profiler import profile
import time

GPT_PAT = r"""'(?:[sdmt]|ve|re|ll)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT_REGEX = regex.compile(GPT_PAT)

# examples: 
# am/are will would have is
# I don't. I'm. I'd. I've. You're. You'll. You've. It's
    
    
@dataclass(order=False)
class BytePair:
    pair: tuple[int, int]
    compare_key: tuple[bytes, bytes]
    freq: int
    
    def __lt__(self, other: "BytePair"):
        return (self.freq, self.compare_key) < (other.freq, other.compare_key)
    
# now we need the BPE algorithm

@dataclass
class PairMetaData:
    freq: int = 0
    nodes: list = field(default_factory=list)

class BPETrainer:
    def __init__(self, input_path: str, vocab_size: int, special_tokens: list[str]):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        
    
    def _find_chunk_boundaries_parallel(self, mm: mmap.mmap, desired_chunks: int = 32, n_jobs: int = 16) -> list[int]:
        """
        We have a file, desired chunks and special token
        
        First verify special token is indeed made of bytes (type system)
        
        Then get the filesize. This allows us to figure out our rough boundaries since we know filesize bytes and desired_chunks. boundaries is desired_chunks + 1
        bytes per chunk is filesize // desired_chunks for most, and a little more at the end
        
        Then iterate through the inner boundaries 1 ~ last - 1
        Start from greater of that position and previous special_token spot. Seek file to there. Read bytes in chunks and seek for the special token
        When parallelizing we do the same thing but we use multicore and only search within the chunk. 
        
        """
        assert isinstance(self.special_tokens, list) and all(isinstance(special_tok, str) for special_tok in self.special_tokens), "The provided special token must be a list of strings"
        
        # init extra vars
        
        
        # find filesize
        mm.seek(0, os.SEEK_END)
        filesize = mm.tell()
        
        chunk_size = filesize // desired_chunks
        boundaries = [i * chunk_size for i in range(desired_chunks + 1)]
        boundaries[-1] = filesize
        
        # define the shared object (mmap file)
        # need byte variant of special_tokens for comparison to bytestring
        special_tokens = [s.encode('utf-8') for s in self.special_tokens]
        shared_objects = (mm, special_tokens, filesize)
        
        def task(shared_objects: tuple[mmap.mmap, bytes, int], start: int, end: int):
            mm, special_tokens, filesize = shared_objects
            
            mini_chunk_size = 4096
            
            mm.seek(start)
            
            cur_pos = start
            
            while cur_pos < end:
                mini_chunk = mm.read(mini_chunk_size)
                if mini_chunk == b"":
                    return filesize #at end this is the bound
                
                for special_tok in special_tokens:
                    special_at = mini_chunk.find(special_tok)
                    # if any match break
                    if special_at != -1:
                        return cur_pos + special_at # found special  point
                
                cur_pos += mini_chunk_size # continue
                
            # if past end we have defined for this chunksize. arbitrarily set to end of file. if some other one exists next chunk will find it
            # note potential error if special tok only straddle edges but will make do for parallelization purposes
            return filesize
            
        
        with WorkerPool(n_jobs=n_jobs, shared_objects=shared_objects, start_method="fork") as pool:
            results = pool.map(task, [(boundaries[i], boundaries[i + 1]) for i in range(1, len(boundaries) - 1)])
            #only move the edge between 2 chunks. hence lose the first left and final right borders. readd at end
            
        return sorted(set([0, *results, filesize]))
    

    def _build_pretoken_counts(self, mm: mmap.mmap, boundaries: list[int], regex_PAT: regex.Pattern = GPT_REGEX, n_jobs: int = 16, profile=False) -> Counter[bytes]:
        """
        Takes in a path or string to file and special token. 
        Then pretokenizes using the provided regex pattern. 
        
        The way we do this in parallel across CPU cores is to take the boundaries and feed it in as a list into the worker. then pretokenize 
        each chunk independently since we do not cut across any pretoken. we still need to take each chunk and split on pretokens. 
        
        Then pretokenize each independently
        """
        
        t0 = time.perf_counter()
        
        boundaries_tuples = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
        
        byte_pattern = b'|'.join([regex.escape(s.encode('utf-8')) for s in self.special_tokens])
        split_PAT = regex.compile(byte_pattern)
        
        shared_objects = (mm, split_PAT)
        
        def task(shared_objects: tuple[mmap.mmap, regex.Pattern], start: int, end: int):
            if profile:
                import cProfile
                inner = cProfile.Profile()
                inner.enable()
            mm, split_PAT = shared_objects
            read_in = end - start #read in from start not up to end since right bound. start of next chunk
            
            # seek out the start 
            mm.seek(start)
            
            # read in the amount of bytes in the chunk
            chunk = mm.read(read_in)
            
            # now split by our special token
            sub_chunks = regex.split(split_PAT, chunk)
            
            base: Counter[bytes] = Counter()
            
            for sub_chunk in sub_chunks: # convert to a map subsequently
                match_iterator = regex.finditer(regex_PAT, sub_chunk.decode("utf-8"))
                base.update([m.group().encode("utf-8") for m in match_iterator]) # convert to string for regex first
            
            if profile:
                inner.disable()
                inner.dump_stats(f"profile_{os.getpid()}_{start}_{end}.prof")
                    
            return base
        
        pretoken_counts: Counter[bytes] = Counter()
        
        with WorkerPool(n_jobs=n_jobs, shared_objects=shared_objects, start_method='fork') as Pool:
            chunked_pretoken_counts: list[Counter[bytes]] = Pool.map(task, boundaries_tuples, progress_bar=True, progress_bar_options={
                "desc": "Counting pretokens..."
            })
            
            
        for _pretoken_counts in chunked_pretoken_counts:
            pretoken_counts.update(_pretoken_counts)
            
        print("Wall: ", time.perf_counter() - t0)
            
        return pretoken_counts  
    
    def _prepare_pointers(self, ref_pretoken_counts: Counter[bytes], vocab: dict[int, bytes]):
        # take the pretoken_counts Counter dict. Take all the keys and convert to byte pairs
        pair_datas: defaultdict[tuple[int, int], PairMetaData] = defaultdict(PairMetaData)
        pretoken_ids: list[int] = []
        pretoken_counts: list[int] = [] # pretoken id (index) -> Count
        
        # each node represent a particular merged tok. And points to its pretoken ID (for quick ref to counts). Next nodes, and prevs
        nodes: list[int] = []
        nexts: list[int] = []
        prevs: list[int] = []
        
        node_id = 0
        for pretok_id, (pretoken, pretok_count) in enumerate(ref_pretoken_counts.items()):
            pretoken_counts.append(pretok_count) #ID is index.
            
            for i, byte_int in enumerate(pretoken):
                pretoken_ids.append(pretok_id)
                nodes.append(byte_int) #indexed byte so its an int
                prevs.append(-1 if i == 0 else node_id - 1)
                if i < len(pretoken) - 1:
                    # build pair
                    pair = (byte_int, pretoken[i + 1]) #bytestring single index returns int 0-255
                    pair_datas[pair].freq += pretok_count # this pair exist in pretok. mul by pretok count
                    pair_datas[pair].nodes.append(node_id) 
                    nexts.append(node_id + 1)
                else:
                    nexts.append(-1) # last one points nowhere
                    
                node_id += 1
                
        # now that we have all the pairs build max_pairs
        max_pairs: list[tuple[int, tuple[bytes, bytes]]] = [
            BytePair((lid, rid), (vocab[lid], vocab[rid]), data.freq) for (lid, rid), data in pair_datas.items()
        ] # max heap
        
        heapq.heapify_max(max_pairs)
        
        return (max_pairs, pair_datas, pretoken_counts, pretoken_ids, nodes, nexts, prevs)
    
    @profile
    def _merge(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], max_pairs: list[BytePair], pair_datas: defaultdict[tuple[int, int], PairMetaData], pretoken_counts: list[int], pretoken_ids: list[int], nodes: list[int], nexts: list[int], prevs: list[int]):
        while True:
            max_pair: BytePair = heapq.heappop_max(max_pairs)
            if (max_pair.freq != 0 and max_pair.freq == pair_datas[max_pair.pair].freq):
                break # not stale! can break
        
        pair_data = pair_datas[max_pair.pair]
        next_tok_id = len(vocab)
        max_pair_lhs, max_pair_rhs = max_pair.pair
        merged_bytes = vocab[max_pair_lhs] + vocab[max_pair_rhs]
        merges.append((vocab[max_pair_lhs], vocab[max_pair_rhs]))
        vocab[next_tok_id] = merged_bytes
        
        valid = 0 
        stale = 0
            
        # now we have our non-stale pair. Find the node
        changed_pairs: set[tuple[int, int]] = set()
        node_ids = pair_data.nodes
        for node_id in node_ids:
            t2 = time.perf_counter()
            pretoken_id = pretoken_ids[node_id]
            pretoken_count = pretoken_counts[pretoken_id]
            next_id = nexts[node_id]
            
            # pair not valid (as stale pairs can exist)
            if (next_id == -1):
                stale += 1
                continue
            
            next_next_id = nexts[next_id] #next must exist by pairness
            prev_id = prevs[node_id]
            
            #pair stale
            if(max_pair_lhs != nodes[node_id] or max_pair_rhs != nodes[next_id]):
                stale += 1
                continue # if stale skip
            
            valid += 1
            
            #now we have a,b,c,d.
            #note the new pair is merged to start point, so a does not need to be repointed but it must be updated
            if (prev_id != -1): #exists something before our pair
                lid, rid = nodes[prev_id], nodes[node_id]
                prev_pair = (lid, rid)
                pair_datas[prev_pair].freq -= pretoken_count
                # we keep track of all the changed pairs to update the heap all at once. since a particular merge only affects the (lhs, rhs). The new pairs formed are not ran through thus can be deferred
                changed_pairs.add(prev_pair) #pair_datas will keep the actual count
                # now add the new pair a, (b,c)
                lid, rid = nodes[prev_id], next_tok_id
                new_pair = (lid, rid)
                pair_datas[new_pair].freq += pretoken_count
                pair_datas[new_pair].nodes.append(prev_id)
                changed_pairs.add(new_pair)
                
            if (next_next_id != -1): #exists something after our pair
                #now we handle right side. c-d. First remove the c-d pair
                lid, rid = nodes[next_id], nodes[next_next_id]
                prev_pair = (lid, rid)
                pair_datas[prev_pair].freq -= pretoken_count
                changed_pairs.add(prev_pair)
                
                #Add the bc - d pair
                lid, rid = next_tok_id, nodes[next_next_id]
                new_pair = (lid, rid)
                pair_datas[new_pair].freq += pretoken_count
                pair_datas[new_pair].nodes.append(node_id)
                changed_pairs.add(new_pair)
            
                # remove the c - d links
                prevs[next_next_id] = node_id # d is now linked to b
            # remove the c - d links
            nexts[next_id] = -1 # c is gone
            
            # remove the b-c links
            prevs[next_id] = -1
            nexts[node_id] = next_next_id # b now points to d
            
            # now we can finally change b to be bd. note we do this later to properly update pair counts before changing. But could just also keep the previous node_value
            nodes[node_id] = next_tok_id
            
        #update max_heap
        for pair in changed_pairs:
            if pair_datas[pair].freq <= 0: continue
            lid, rid = pair
            heapq.heappush_max(max_pairs, BytePair(pair, (vocab[lid], vocab[rid]), pair_datas[pair].freq))
        
        #normally we would want to remove all of bc as well. but we removed all of them via the merge so just set to 0. no need to update queue
        pair_data.freq = 0
        pair_data.nodes.clear()
    
    def train(self, num_chunks, profile=False, pretoken_count_file: pathlib.Path | str = None, save = False, pretoken_count_save_file: pathlib.Path | str = None):
        if save:
            assert pretoken_count_save_file is not None, "You must have a save output file for pretoken counts if you want to save"
        #base vocab size is 256 (for english) ASCII UTF-8 Bytes
        
        if pretoken_count_file is None:
            with open(self.input_path) as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                
                boundaries = self._find_chunk_boundaries_parallel(mm, num_chunks)
                print("finished building boundaries")
                self.pretoken_counts = self._build_pretoken_counts(mm, boundaries, profile=profile)
                print("built pretoken counts")
                
                if save:
                    with open(pretoken_count_save_file, "wb") as pf:
                        pickle.dump(self.pretoken_counts, pf)
                        
        else:
            with open(pretoken_count_file, "rb") as pf:
                self.pretoken_counts = pickle.load(pf)
                
        (vocab, merges) = rust_module.prepare_and_train_merges(self.pretoken_counts, self.special_tokens, self.vocab_size)
        
        # import cProfile
        # profiler = cProfile.Profile()
        # profiler.enable()
            
        # profiler.disable()
        # stats = pstats.Stats(profiler)
        # stats.sort_stats("cumtime")
        # stats.print_stats(30)
            
        # save to object
        self.vocab = vocab
        self.merges = merges
            
        return (vocab, merges)
