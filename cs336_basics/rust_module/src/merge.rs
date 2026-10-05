use pyo3::prelude::*;
use pyo3::types::*;
use std::collections::HashMap;
use std::collections::BinaryHeap;
use std::collections::HashSet;
use indicatif::ProgressBar;
use rayon::prelude::*;
use aho_corasick::{AhoCorasick, MatchKind};
use regex::Regex;

const GPT2_SPLITTER_TEXT: &str =
r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+"; // contains everything except for the whitespace work

// rust compares top down so this is valid
#[derive(Eq, PartialEq, Ord, PartialOrd)]
pub struct BytePair {
    freq: i32,
    compare_key: (Vec<u8>, Vec<u8>),
    key: usize,
}

#[derive(Default, Debug)]
pub struct PairMetaData {
    freq: i32,
    nodes: Vec<usize>
}

fn _prepare(map: &HashMap<Vec<u8>, i32>, vocab: &HashMap<usize, Vec<u8>>) -> (BinaryHeap<BytePair>, HashMap<usize, PairMetaData>, Vec<i32>, Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut pair_datas: HashMap<usize, PairMetaData> = HashMap::new();

    let mut pretoken_ids: Vec<usize> = Vec::new();
    let mut pretoken_counts: Vec<i32> = Vec::new();

    // each node represent a particular merged tok. And points to its pretoken ID (for quick ref to counts). Next nodes, and prevs
    let mut nodes: Vec<usize> = Vec::new(); // tokID
    let mut nexts: Vec<usize> = Vec::new();
    let mut prevs: Vec<usize> = Vec::new();

    let mut node_id: usize = 0;

    for (pretok_id, (pretoken, pretoken_count)) in map.iter().enumerate() {
        pretoken_counts.push(*pretoken_count);

        for i in 0..pretoken.len() {
            pretoken_ids.push(pretok_id as usize);
            let byte_int = pretoken[i] as usize;

            nodes.push(byte_int);
            prevs.push(if i == 0 {usize::MAX} else {node_id - 1}); // since using usize (MAX) means invalid now

            if i < pretoken.len() - 1 { // before last
                nexts.push(node_id + 1);
                // build pair (a,b) as single int
                let pair_key: usize = (byte_int << 32) | (pretoken[i + 1] as usize); // tok key | tok key (both usize but shifted)
                let entry = pair_datas.entry(pair_key).or_default();
                entry.freq += *pretoken_count;
                entry.nodes.push(node_id);
            } else {
                nexts.push(usize::MAX); // last one points nowhere
            }
            node_id += 1;
        }
    }

    let max_pairs: BinaryHeap<BytePair> = pair_datas
        .iter()
        .map(|(&key, data)| {
            let lid = key >> 32;
            let rid = key & 0xFFFF_FFFF;
            BytePair {
                freq: data.freq,
                compare_key: (vocab[&lid].clone(), vocab[&rid].clone()),
                key,
            }
        })
        .collect();
    
    return (max_pairs, pair_datas, pretoken_counts, pretoken_ids, nodes, nexts, prevs)

}

fn _train_merges(num_merges: usize, mut max_pairs: BinaryHeap<BytePair>, mut pair_datas: HashMap<usize, PairMetaData>, pretoken_counts: &[i32], pretoken_ids: &[usize], mut nodes: Vec<usize>, mut nexts: Vec<usize>, mut prevs: Vec<usize>, vocab: &mut HashMap<usize, Vec<u8>>, merges: &mut Vec<(Vec<u8>, Vec<u8>)>) {
    let bar: ProgressBar = ProgressBar::new(num_merges as u64);
    for _ in 0..num_merges {
        let max_pair = loop {
            let Some(max_pair) = max_pairs.pop() else {
                break None;
            };
    
            if max_pair.freq != 0 && max_pair.freq == pair_datas[&max_pair.key].freq {
                break Some(max_pair); // not stale
            }
        };
    
        let Some(max_pair) = max_pair else {
            return
        };
    
        let node_ids = pair_datas[&max_pair.key].nodes.clone();
        let next_tok_id = vocab.len();
        let max_pair_lid = max_pair.key >> 32;
        let max_pair_rid = max_pair.key & 0xFFFF_FFFF;
        let mut merged_bytes = vocab[&max_pair_lid].clone();
        merged_bytes.extend(&vocab[&max_pair_rid]); // extend with rid
        
        merges.push((vocab[&max_pair_lid].clone(), vocab[&max_pair_rid].clone()));
        vocab.insert(next_tok_id, merged_bytes.clone());
    
        let mut changed_pairs: HashSet<usize> = HashSet::new();
        for &node_id in &node_ids {
            let orig_node_id = nodes[node_id];
            let pretoken_id = pretoken_ids[node_id];
            let pretoken_count = pretoken_counts[pretoken_id];
            let next_id = nexts[node_id];
    
            if next_id == usize::MAX {
                continue;
            }
    
            let next_next_id = nexts[next_id];
            let prev_id = prevs[node_id];
    
            if max_pair_lid != nodes[node_id] || max_pair_rid != nodes[next_id] {
                continue;
            }
    
            if prev_id != usize::MAX {
                let key = (nodes[prev_id] << 32) | orig_node_id;
                pair_datas.entry(key).or_default().freq -= pretoken_count;
                changed_pairs.insert(key);
    
                // add new pair
                let key = nodes[prev_id] << 32 | next_tok_id;
                let entry = pair_datas.entry(key).or_default();
                entry.freq += pretoken_count;
                entry.nodes.push(prev_id);
                changed_pairs.insert(key);
            }
    
            if next_next_id != usize::MAX {
                // prev key
                let key = (nodes[next_id] << 32) | nodes[next_next_id];
                pair_datas.entry(key).or_default().freq -= pretoken_count;
                changed_pairs.insert(key);
    
                let key = next_tok_id << 32 | nodes[next_next_id];
                let entry = pair_datas.entry(key).or_default();
                entry.freq += pretoken_count;
                entry.nodes.push(node_id);
                changed_pairs.insert(key);
    
                // d should now point to b
                prevs[next_next_id] = node_id;
            }
    
            // c no longer exists cut outgoing
            nexts[next_id] = usize::MAX;
            prevs[next_id] = usize::MAX;
    
            // replace b -> c link with b -> d (if it DNS (usize::MAX) still a valid target from here)
            nexts[node_id] = next_next_id; // note if next_next_id exists will alr have been set. we dont do it here since usize::MAX is an invalid index
    
            // now we can set value of node to true value
            nodes[node_id] = next_tok_id;
        }
    
        // since we removed all pairs. set freq to 0 and erase nodes
        let entry = pair_datas.entry(max_pair.key).or_default();
        entry.freq = 0;
        entry.nodes.clear();
    
        // now update heap with all the changed values
        for changed_pair in changed_pairs {
            let pair_data = &pair_datas[&changed_pair];
            let lid = changed_pair >> 32;
            let rid = changed_pair & 0xFFFF_FFFF;
            max_pairs.push(
                BytePair {
                    freq: pair_data.freq,
                    compare_key: (vocab[&lid].clone(), vocab[&rid].clone()),
                    key: changed_pair
                }
            )
        }
        bar.inc(1);
    }

    bar.finish();
    
}

#[pyfunction]
pub fn prepare_and_train_merges(pretoken_counts: HashMap<Vec<u8>, i32>  , special_tokens: Vec<String>, vocab_size: usize) -> (HashMap<usize, Vec<u8>>, Vec<(Vec<u8>, Vec<u8>)>) {
    let mut vocab: HashMap<usize, Vec<u8>> = HashMap::new();
    let mut merges: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    for i in 0..=255 {
        vocab.insert(i as usize, vec![i]);
    }

    for special_tok in special_tokens {
        vocab.insert(vocab.len(), special_tok.as_bytes().to_vec());
    }

    let num_merges = vocab_size - vocab.len();

    let (max_pairs, pair_datas, pretoken_counts, pretoken_ids, nodes, nexts, prevs) = _prepare(&pretoken_counts, &vocab);
    _train_merges(num_merges, max_pairs, pair_datas, &pretoken_counts, &pretoken_ids, nodes, nexts, prevs, &mut vocab, &mut merges);

    return (vocab, merges); // move out
}

#[pyclass]
struct EncodeIter {
    tokenizer: Py<RustTokenizer>,
    source: Py<PyIterator>,
    pending: std::vec::IntoIter<usize>
}

#[pymethods]
impl EncodeIter {
    fn __iter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<usize>> {
        loop {
            if let Some(id) = self.pending.next() {
                return Ok(Some(id));
            }

            let Some(item) = self.source.bind(py).clone().next() else {
                return Ok(None);
            };

            let text: String = item?.extract()?;
            let ids = self.tokenizer.borrow(py).encode(&text);

            self.pending = ids.into_iter();
        }
    }
}

#[pyclass]
pub struct RustTokenizer {
    vocab: HashMap<usize, Vec<u8>>,
    reverse_vocab: HashMap<Vec<u8>, usize>,
    merges: Vec<(usize, usize, usize)>,
    ac: Option<AhoCorasick>,
    re: Regex
}

#[pymethods]
impl RustTokenizer {
    #[new]
    pub fn new(vocab: HashMap<usize, Vec<u8>>, merges: Vec<(Vec<u8>, Vec<u8>)>, special_tokens: Option<Vec<String>>, regex_pat_provided: Option<&str>) -> Self {
        // build reverse vocab
        let reverse_vocab: HashMap<Vec<u8>, usize> = vocab.iter().map(|(&id, bytes)| (bytes.clone(), id)).collect();
        let mut int_merges: Vec<(usize, usize, usize)> = Vec::new();
        for merge in merges {
            let mut left_bytes = merge.0;
            let left_id = reverse_vocab[&left_bytes];
            let right_bytes = merge.1;
            let right_id = reverse_vocab[&right_bytes];
            left_bytes.extend(right_bytes); // get joint bytes 
            let joint_id = reverse_vocab[&left_bytes];

            int_merges.push((left_id, right_id, joint_id));
        }

        let mut ac: Option<AhoCorasick> = None;

        if let Some(special_toks) = &special_tokens {
            ac = Some(AhoCorasick::builder().match_kind(MatchKind::LeftmostLongest).build(special_toks).unwrap());
        }

        let regex_str: &str = regex_pat_provided.unwrap_or(GPT2_SPLITTER_TEXT);
        let regex_pat: Regex = Regex::new(regex_str).unwrap();

        Self {vocab, reverse_vocab, merges: int_merges, ac, re:regex_pat}
    }

    #[classmethod]
    pub fn from_file(_cls: &Bound<'_, PyType>, py: Python<'_>, vocab_filepath: &str, merges_filepath: &str, special_tokens: Option<Vec<String>>, regex_pat_provided: Option<&str>) -> PyResult<Self> {
        let pickle = py.import("pickle")?;
        let builtins = py.import("builtins")?;
        // load in from file

        let vocab_file = builtins.call_method1("open", (vocab_filepath, "rb"))?;
        let vocab_obj = pickle.call_method1("load", (&vocab_file,))?;

        let merges_file = builtins.call_method1("open", (merges_filepath, "rb"))?;
        let merges_obj = pickle.call_method1("load", (&merges_file,))?;

        let vocab: HashMap<usize, Vec<u8>> = vocab_obj.extract()?;
        let merges: Vec<(Vec<u8>, Vec<u8>)> = merges_obj.extract()?;

        Ok(Self::new(vocab, merges, special_tokens, regex_pat_provided))
    }

    pub fn decode(&self, ids: Vec<usize>) -> PyResult<String> {
        // flat_map gets us reference to id. use to get reference to bytes. Copy so we dont destroy vocab and collect
        let bytes: Vec<u8> = ids.iter().flat_map(|id| &self.vocab[id]).copied().collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned()) // result since conversion may fail
    }

    pub fn encode(&self, text:&str) -> Vec<usize> {
        let (pretoken_chunks, special_tok_chunks) = self._pretokenize(text);
        self._prepare_and_run_merges(pretoken_chunks, special_tok_chunks)
    }

    fn encode_iterable(
        slf: Py<Self>,
        iterable: &Bound<'_, PyAny>,
    ) -> PyResult<EncodeIter> {
        Ok(EncodeIter {
            tokenizer: slf,
            source: iterable.try_iter()?.unbind(),
            pending: Vec::new().into_iter(),
        })
    }
}

impl RustTokenizer {
    pub fn _pretokenize<'a>(&self, text: &'a str) -> (Vec<Vec<&'a [u8]>>, Vec<&'a [u8]>) {
        let mut subtexts: Vec<&str> = Vec::new();
        let mut pretokens_special: Vec<&[u8]> = Vec::new();
        if let Some(ac) = &self.ac {
            let mut last = 0;
            for mat in ac.find_iter(text) {
                let normal = &text[last..mat.start()];
                let special = &text[mat.start()..mat.end()];

                subtexts.push(normal);
                pretokens_special.push(special.as_bytes());

                last = mat.end(); // start of next pretoken (or where match ends)
            }
            // after match there may be another token account for this
            let trailing = &text[last..];
            subtexts.push(trailing);
        } else {
            subtexts.push(text);
        }

        // we are now ready to pretokenize using custom rust regex -> direct u8 bytes
        let pretokens_non_special: Vec<Vec<&[u8]>> = subtexts.par_iter().map(|subtext| {
            let mut last_end = 0;
            let mut pretokens_non_special_chunk: Vec<&[u8]> = Vec::new();
            for mat in self.re.find_iter(subtext) {
                let mat_bytes = mat.as_str().as_bytes();
                let start = mat.start();
                // always check for whitespace before cur token (means we have to add whitespace token). Note how we add whitespace depends on next token so add whitespace before token
                if last_end < start { // means some extra tokens here
                    if mat_bytes.first() == Some(&b' ') { // is a space then the entire whitespace before is pretoken
                        pretokens_non_special_chunk.push(subtext[last_end..start].as_bytes())
                    } else {
                        // the long whitespace and the final char
                        let last_char_size = subtext[last_end..start].chars().next_back().unwrap().len_utf8();
                        if last_char_size < start - last_end { // more than last_char
                            pretokens_non_special_chunk.push(subtext[last_end..start-last_char_size].as_bytes());
                        }
                        pretokens_non_special_chunk.push(subtext[start - last_char_size..start].as_bytes());
                    }
                }
                pretokens_non_special_chunk.push(mat_bytes); // now we can add pretoken
                last_end = mat.end();
            }
            if last_end < subtext.len() {
                // check for trailing
                let trailing = subtext[last_end..].as_bytes();
                pretokens_non_special_chunk.push(trailing);
            }
            return pretokens_non_special_chunk;
        }).collect();

        return (pretokens_non_special, pretokens_special)
    }

    pub fn _prepare_and_run_merges(&self, pretoken_chunks: Vec<Vec<&[u8]>>, special_tok_chunks: Vec<&[u8]>) -> Vec<usize> {
        let mut nodes: Vec<usize> = Vec::new();
        let mut nexts: Vec<usize> = Vec::new();
        let mut prevs: Vec<usize> = Vec::new();
        let mut pair_datas: HashMap<usize, PairMetaData> = HashMap::new();
        
        let mut unique_pretokens_to_start_node: HashMap<&[u8], usize> = HashMap::new();
        for pretoken_chunk in &pretoken_chunks {
            for pretoken in pretoken_chunk {
                if !unique_pretokens_to_start_node.contains_key(*pretoken) {
                    unique_pretokens_to_start_node.insert(*pretoken, usize::MAX);
                }
            }
        }
        
        let mut node_id: usize = 0;
        let pretokens: Vec<&[u8]> = unique_pretokens_to_start_node.keys().copied().collect();
        for pretoken in pretokens {
            for i in 0..pretoken.len() {
                let vocab_id = self.reverse_vocab[&pretoken[i..i+1]]; // convert to vocabID for byte
                nodes.push(vocab_id as usize);
                // at each node point at neighbors if exists
                prevs.push(if i==0 {usize::MAX} else {node_id - 1});
    
                if i == 0 {
                    *unique_pretokens_to_start_node.get_mut(pretoken).unwrap() = node_id;
                }
    
                if i < pretoken.len() - 1 {
                    nexts.push(node_id + 1);
                    let next_tok_id = self.reverse_vocab[&pretoken[i+1..i+2]]; // = [next_byte]
                    let pair_key: usize = (vocab_id << 32) | (next_tok_id);
                    let entry = pair_datas.entry(pair_key).or_default();
                    entry.nodes.push(node_id);
                } else {
                    nexts.push(usize::MAX);
                }
    
                node_id += 1;
            }
        }
    
        // eprint!("{:#?}", pair_datas);
    
        // now merge through the merges
        for merge in &self.merges {
            let pair_key = merge.0 << 32  | merge.1;
            let Some(pair_data) = pair_datas.remove(&pair_key) else { // we only do a merge once so can remove this pair_key 
                continue;
            };
            let tok_id = merge.2;
            for node_id in pair_data.nodes {
                // check staleness
                let next_id = nexts[node_id];
                if next_id == usize::MAX {
                    continue;
                }
    
                let next_next_id = nexts[next_id];
                let prev_id = prevs[node_id];
    
                if (nodes[node_id] != merge.0) || (nodes[next_id] != merge.1) {
                    continue; // also stale pair does not match atp
                }
    
                if prev_id != usize::MAX {
                    // remove a-b link. add a-bc. dont change frequencies, dont delete nodes (soft delete so no vector reordering)
                    let new_key = nodes[prev_id] << 32 | tok_id;
                    pair_datas.entry(new_key).or_default().nodes.push(prev_id); // now (a,bc)
                    // pointers already set. 
                }
    
                if next_next_id != usize::MAX {
                    let new_key = tok_id << 32 | nodes[next_next_id];
                    pair_datas.entry(new_key).or_default().nodes.push(node_id); // new pair forms here (bc, d)
    
                    // point d to b
                    prevs[next_next_id] = node_id;
                }
    
                // remove all c pointers
                prevs[next_id] = usize::MAX;
                nexts[next_id] = usize::MAX;
    
                // point b -> d regardless
                nexts[node_id] = next_next_id;
    
                // can finally swap node_id value to true token id
                nodes[node_id] = tok_id;
            }
        }
    
        // once done loop through the pretoken_start_nodes to get the replacement for each pretoken in tokenIDs
        let mut pretoken_to_ids: HashMap<&[u8], Vec<usize>> = HashMap::new();
    
        // pretoken_start_nodes correlates to each unique pretoken (same order)
        for (pretoken, start_node) in unique_pretokens_to_start_node {
            let mut one_pretoken_ids: Vec<usize> = Vec::new();
            let mut node_id = start_node;
            while node_id != usize::MAX {
                one_pretoken_ids.push(nodes[node_id]);
                node_id = nexts[node_id];
            }
            pretoken_to_ids.insert(pretoken, one_pretoken_ids);
        } // all pretoken ids for each pretoken. Now reinsert to the original pretoken chunks and merge with special_toks
    
        let mut final_tok_ids: Vec<usize> = Vec::new();
        for (i, pretoken_chunk) in pretoken_chunks.iter().enumerate() {
            for pretoken in pretoken_chunk {
                final_tok_ids.extend(&pretoken_to_ids[pretoken]);
            }
            // first pretoken is followed by the first special_tok. each token (even) empty has a special except for last one. 
            if i < special_tok_chunks.len() {
                final_tok_ids.push(self.reverse_vocab[special_tok_chunks[i]])
            }
            // special_tok_chunk is always i+1 except for last i since inbetween any real tok chunk
        }
    
        return final_tok_ids;
    }
}