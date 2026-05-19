from locale import normalize
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

byte_encoder = bytes_to_unicode()

def normalize_MiRe(text):
    return " ".join(["".join(byte_encoder[b] for b in x.encode("utf-8")) for x in text.split()])