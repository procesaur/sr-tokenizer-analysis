from datasets import load_dataset
from ..conversion import convert


def preprocess(example):
    return {"text": convert(example["text"])}


dataset = load_dataset("procesaur/sr-tokenizer-test", split="train", keep_in_memory=False)
dataset = dataset.map(preprocess)
dataset.save_to_disk("./serbian_tokenizer_dataset")

dataset = load_dataset("procesaur/sr-tokenizer-test", split="test", keep_in_memory=False)
dataset = dataset.map(preprocess)
dataset.save_to_disk("./serbian_tokenizer_dataset_test")
