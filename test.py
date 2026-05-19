from transformers import PreTrainedTokenizerFast

# Load tokenizer from JSON file
tokenizer = PreTrainedTokenizerFast(tokenizer_file="sample_tokenizers/MiRe_bpe.json")

# Encode a string → token IDs
text = "krivičnim"
ids = tokenizer.encode(text)
print("Token IDs:", ids)

# Decode back → fully readable text
decoded = [tokenizer.decode(id) for id in ids]
print("Decoded text:", decoded)
