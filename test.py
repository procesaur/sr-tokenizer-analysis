from transformers import PreTrainedTokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

byte_encoder = bytes_to_unicode()
# Load tokenizer from JSON file
tokenizer = PreTrainedTokenizerFast(tokenizer_file="sample_tokenizers/MiRe_bpe.json")
#tokenizer = PreTrainedTokenizerFast(tokenizer_file="sample_tokenizers/MorfoTok.json")
# Encode a string → token IDs
text = "krivičnim delom"

text = ["".join(byte_encoder[b] for b in x.encode("utf-8")) for x in text.split()]

ids = tokenizer.encode(text)
print("Token IDs:", ids)

# Decode back → fully readable text
decoded = [tokenizer.decode(id) for id in ids]
print("Decoded text:", decoded)
