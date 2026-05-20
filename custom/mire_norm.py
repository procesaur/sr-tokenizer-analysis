#from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    Useful for BPE tokenization since we want to avoid mapping to whitespace/control characters.
    """
    bs = list(range(ord("!"), ord("~")+1)) + \
         list(range(ord("¡"), ord("¬")+1)) + \
         list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


byte_encoder = bytes_to_unicode()

def normalize_MiRe(text):
    return " ".join(["".join(byte_encoder[b] for b in x.encode("utf-8")) for x in text.split()])