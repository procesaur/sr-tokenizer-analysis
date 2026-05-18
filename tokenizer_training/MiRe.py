import numpy as np
from collections import Counter
from multiprocessing import Pool
from tqdm import tqdm
from json import load, dump
from tokenizers import Tokenizer, AddedToken


def create_added_token(token):
    return AddedToken(
        token, 
        normalized=False,  # CRITICAL: Prevents pre-processing/lowercasing/spacing changes
        special=True,      # CRITICAL: Keeps it completely atomic during tokenization splits
        single_word=False   # Ensures it acts as an un-splittable block
    )

tokenizer = None
end_suffix = "Ġ"

def process_example(example):
    local_counter = Counter()
    whitespace_list = example["text"].split(" ")
    for item in whitespace_list:
        encoded = tokenizer.encode(item)
        tokens = encoded.tokens
        seq = []
        for tok in tokens:
            if len(tok) == 1 and tok != end_suffix:
                seq.append(tok)
            else:
                if tok == end_suffix:
                    seq.append(tok)
                if len(seq) > 1:
                    local_counter["".join(seq)] += 1
                seq = []
        if len(seq) > 1:
            local_counter["".join(seq)] += 1
    return local_counter


def process_example2(example):
    local_counter = Counter()
    whitespace_list = example["text"].split(" ")
    for item in whitespace_list:
        encoded = tokenizer.encode(item)
        tokens = encoded.tokens
        for tok in tokens:
            local_counter[tok] += 1
    return local_counter


def init_worker(tokenizerx, suffix):
    global tokenizer, end_suffix
    tokenizer = tokenizerx
    end_suffix = suffix


def get_measures(counter: Counter):  # Returns H (entropy) and R (redundancy)
    freq = np.array(list(counter.values()), dtype=np.float64)
    p = freq / freq.sum()
    len_p = len(p)
    H = -(p*np.log2(p)).sum() 
    R = 1-(H/np.log2(len_p))
    return  H, R


def create_base_tokenizer(cutoff, latin=True):

    if latin:
        with open("sample_tokenizers/ts_bpe.json", "r", encoding="utf-8") as f:
            data = load(f)
    else:
        with open("sample_tokenizers/ts_bpe_c.json", "r", encoding="utf-8") as f:
            data = load(f)

    original_vocab = data["model"]["vocab"]
    sorted_vocab = sorted(original_vocab.items(), key=lambda x: x[1])
    new_vocab = {token: idx for token, idx in sorted_vocab[:cutoff]}
    data["model"]["vocab"] = new_vocab
    if "merges" in data["model"]:
        data["model"]["merges"] = []

    with open("temp.json", "w", encoding="utf-8") as f:
        dump(data, f, ensure_ascii=False)

    return Tokenizer.from_file("temp.json")
    
    #vocab = tokenizer.get_vocab()
    #extra_tokens = [tok for tok, idx in vocab.items()]
    #added_tokens = [create_added_token(tok) for tok in extra_tokens]
    #tokenizer.add_tokens(added_tokens)


def update_tokens_from_count(tokenizer, counter, vocab_size):
    scores = {seq: freq * len(seq) for seq, freq in counter.items()}
    top_sequences = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:vocab_size]
    new_tokens = [create_added_token(seq) for seq, _ in top_sequences]
    
    token_iterator = iter(new_tokens)

    with tqdm(total=30000, desc="Adding tokens to vocabulary") as pbar:
        while len(tokenizer.get_vocab()) < vocab_size:
            try:
                token = next(token_iterator)
            except StopIteration:
                print("\nWarning: Exhausted 'new_tokens' list before reaching 30k.")
                break
            added_count = tokenizer.add_tokens([token])
            if added_count > 0:
                pbar.update(1)


def token_freq(dataset, tok, merge=True):
    tokenizer = tok
    counter = Counter()
    if merge:
        with Pool(processes=28, initializer=init_worker, initargs=(tokenizer, end_suffix)) as pool:
            results = list(tqdm(pool.imap(process_example, dataset), total=len(dataset)))
    else:
        with Pool(processes=28, initializer=init_worker, initargs=(tokenizer, end_suffix)) as pool:
            results = list(tqdm(pool.imap(process_example2, dataset), total=len(dataset)))

    for c in results:
        counter.update(c)
    return counter


def HR_inspect(dataset, latin=True, merge=False):
    global tokenizer
    if not merge:
        cutoffs = [261 + i*10 for i in range(50)]
    else:
        cutoffs = [261 + i*50 for i in range(50)]
    Hs = []
    Rs = []
    for x in cutoffs:
        tokenizer = create_base_tokenizer(x, latin)
        freqs = token_freq(dataset, tokenizer, merge)
        H, R = get_measures(freqs)
        Hs.append(H)
        Rs.append(R)
        
    print(Hs)
    print(Rs)
