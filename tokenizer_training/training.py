from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, Regex, AddedToken
from collections import Counter
from tokenization_srna import SrnaTokenizer
from tqdm import tqdm
from json import load, dump
from multiprocessing import Pool


def create_added_token(token):
    return AddedToken(
        token, 
        normalized=False,  # CRITICAL: Prevents pre-processing/lowercasing/spacing changes
        special=True,      # CRITICAL: Keeps it completely atomic during tokenization splits
        single_word=False   # Ensures it acts as an un-splittable block
    )
 
vocab_size = 30000
suffix_vocab_size = 200
latin = True
base_bpe= False
srna = False
original_bpe = True
MiRe_bpe = True
end_suffix = "Ġ"
initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
special_tokens_list = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
boc_token = "<csta>"
eoc_token = "<cend>"
cap_token = "<capi>"
up_token = "<uppe>"
sepcial_tokens_list_srna = ["[PAD]", up_token, cap_token, eoc_token, boc_token]
special_tokens=[create_added_token(x) for x in special_tokens_list]
special_tokens_srna=[create_added_token(x) for x in sepcial_tokens_list_srna]
srnatok = SrnaTokenizer(
    boc_token = boc_token,
    eoc_token = eoc_token,
    cap_token = cap_token,
    up_token = up_token
)
MiRe_cutoff = 768

tokenizer = None


def get_dataset(latin, test=False):
    if latin:
        if test:
            return load_from_disk("./serbian_tokenizer_dataset_test", keep_in_memory=True)
        return load_from_disk("./serbian_tokenizer_dataset", keep_in_memory=True)
    if test:
        return load_dataset(
        "procesaur/sr-tokenizer-test",
        split="train",
        keep_in_memory=True
        )    
    return load_dataset(
        "procesaur/sr-tokenizer-test",
        split="test",
        keep_in_memory=True
        )  


def batch_iterator(dataset, batch_size=10000, fn=None):
    batch = []
    for example in dataset:
        if fn:
            batch.append(fn(example["text"]))
        else:
            batch.append(example["text"])
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def train_bpe(dataset, latin=True):
    tokenizer = Tokenizer(models.BPE(ignore_merges=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.add_special_tokens(special_tokens)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=initial_alphabet
    )
    tokenizer.train_from_iterator(batch_iterator(dataset), trainer=trainer)
    if latin:
        tokenizer.save("tokenizers/bpe.json")
    else:
        tokenizer.save("tokenizers/bpe_c.json")

def srna_prepare(text):
    text = srnatok.prepare_for_tokenization(text)
    for x in sepcial_tokens_list_srna:
        text = text.replace(x, "")
    return text

def train_srna(dataset, latin=True):
    tokenizer = Tokenizer(models.BPE(ignore_merges=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.add_special_tokens(special_tokens_srna)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens_srna,
        initial_alphabet=initial_alphabet
    )
    tokenizer.train_from_iterator(batch_iterator(dataset, fn=srna_prepare), trainer=trainer)
    if latin:
        tokenizer.save("../sample_tokenizers/srna.json")
    else:
        tokenizer.save("../sample_tokenizers/srna_c.json")


def train_original_bpe(dataset, latin=True):
    tokenizer = Tokenizer(models.BPE(ignore_merges=True))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.ByteLevel(add_prefix_space=False)
    ])
    tokenizer.add_special_tokens(special_tokens)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=initial_alphabet,
        end_of_word_suffix=end_suffix
    )
    tokenizer.train_from_iterator(batch_iterator(dataset), trainer=trainer)
    if latin:
        tokenizer.save("../sample_tokenizers/original_bpe.json")
    else:
        tokenizer.save("../sample_tokenizers/original_bpe_c.json")


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


def init_worker(tokenizerx, suffix):
    global tokenizer, end_suffix
    tokenizer = tokenizerx
    end_suffix = suffix


def train_MiRe_bpe(dataset, latin=True):
    global tokenizer
    if latin:
        with open("../sample_tokenizers/original_bpe.json", "r", encoding="utf-8") as f:
            data = load(f)
    else:
        with open("../sample_tokenizers/original_bpe_c.json", "r", encoding="utf-8") as f:
            data = load(f)      



    original_vocab = data["model"]["vocab"]
    sorted_vocab = sorted(original_vocab.items(), key=lambda x: x[1])
    new_vocab = {token: idx for token, idx in sorted_vocab[:MiRe_cutoff]}
    data["model"]["vocab"] = new_vocab
    if "merges" in data["model"]:
        data["model"]["merges"] = []

    with open("temp.json", "w", encoding="utf-8") as f:
        dump(data, f, ensure_ascii=False)

    tokenizer = Tokenizer.from_file("temp.json")

    vocab = tokenizer.get_vocab()
    extra_tokens = [tok for tok, idx in vocab.items()]
    added_tokens = [create_added_token(tok) for tok in extra_tokens]
    tokenizer.add_tokens(added_tokens)

    with Pool(processes=28, initializer=init_worker, initargs=(tokenizer, end_suffix)) as pool:
        results = list(tqdm(pool.imap(process_example, dataset), total=len(dataset)))
    counter = Counter()
    for c in results:
        counter.update(c)

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

    if latin:
        tokenizer.save("../sample_tokenizers/MiRe_bpe.json")
    else:
        tokenizer.save("../sample_tokenizers/MiRe_bpe_c.json")


if __name__ == "__main__":
    for x in [True, False]:
        dataset = get_dataset(latin=x)
        dataset = dataset.select(range(1000))

        train_bpe(dataset, latin=x)
        train_srna(dataset, latin=x)
        train_original_bpe(dataset, latin=x)
        train_MiRe_bpe(dataset, latin=x)
      