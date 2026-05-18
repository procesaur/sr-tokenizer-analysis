from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from custom.tokenization_srna import SrnaTokenizer

from tokenizer_training.MiRe import token_freq, create_base_tokenizer, create_added_token, update_tokens_from_count, HR_inspect, end_suffix


vocab_size = 30000
suffix_vocab_size = 200
latin = True
base_bpe= False
srna = False
ts_bpe = True
MiRe_bpe = True

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


def get_dataset(latin, test=False):
    if latin:
        if test:
            return load_from_disk("tokenizer_training/serbian_tokenizer_dataset_test", keep_in_memory=True)
        return load_from_disk("tokenizer_training/serbian_tokenizer_dataset", keep_in_memory=True)
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
        tokenizer.save("sample_tokenizers/bpe.json")
    else:
        tokenizer.save("sample_tokenizers/bpe_c.json")

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
        tokenizer.save("sample_tokenizers/srna.json")
    else:
        tokenizer.save("sample_tokenizers/srna_c.json")


def train_ts_bpe(dataset, latin=True):
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
        tokenizer.save("sample_tokenizers/ts_bpe.json")
    else:
        tokenizer.save("sample_tokenizers/ts_bpe_c.json")


def train_MiRe_bpe(dataset, latin=True):
    tokenizer = create_base_tokenizer()
    counter = token_freq(dataset, tokenizer)
    tokenizer = update_tokens_from_count(tokenizer, counter, vocab_size)

    if latin:
        tokenizer.save("sample_tokenizers/MiRe_bpe.json")
    else:
        tokenizer.save("sample_tokenizers/MiRe_bpe_c.json")


if __name__ == "__main__":
    latins = [True, False]
    for x in latins:
        dataset = get_dataset(latin=x)
        dataset = dataset.select(range(2000))

        HR_inspect(dataset, x, merge=True)
        HR_inspect(dataset, x, merge=False)

        #train_bpe(dataset, latin=x)
        #train_srna(dataset, latin=x)
        #train_ts_bpe(dataset, latin=x)

        #train_MiRe_bpe(dataset, latin=x)
      