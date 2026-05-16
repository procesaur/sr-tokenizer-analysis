# Copyright 2026 Mihailo Škorić.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for Serbian and Serbo-Croatian models."""


import regex


mapping = {
        "Љ": "Lj", "љ": "lj", "Њ": "Nj", "њ": "nj", "Џ": "Dž", "џ": "dž",
        "А": "A", "а": "a", "Б": "B", "б": "b", "В": "V", "в": "v",
        "Г": "G", "г": "g", "Д": "D", "д": "d", "Ђ": "Đ", "ђ": "đ",
        "Е": "E", "е": "e", "Ж": "Ž", "ж": "ž", "З": "Z", "з": "z",
        "И": "I", "и": "i", "Ј": "J", "ј": "j", "К": "K", "к": "k",
        "Л": "L", "л": "l", "М": "M", "м": "m", "Н": "N", "н": "n",
        "О": "O", "о": "o", "П": "P", "п": "p", "Р": "R", "р": "r",
        "С": "S", "с": "s", "Т": "T", "т": "t", "Ћ": "Ć", "ћ": "ć",
        "У": "U", "у": "u", "Ф": "F", "ф": "f", "Х": "H", "х": "h",
        "Ц": "C", "ц": "c", "Ч": "Č", "ч": "č", "Ш": "Š", "ш": "š",
    }

def cyr2lat(text: str) -> str:
    return "".join(mapping.get(ch, ch) for ch in text)


def lat2cyr(text: str) -> str:
    return "".join({v: k for k, v in mapping.items()}.get(ch, ch) for ch in text)


class SrnaTokenizer():

    def __init__(
        self,
        boc_token = "<|cyr_start|>",
        eoc_token = "<|cyr_end|>",
        cap_token = "<|cap|>",
        up_token = "<|up|>",
        **kwargs,
    ):


        self.boc_token = boc_token
        self.eoc_token = eoc_token
        self.cap_token = cap_token
        self.up_token = up_token

        self.PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
        self.CYRILLIC_REGEX = r"\s?\p{Cyrillic}[^\p{Latin}]*"
        self.CLEANUP_FIND = rf"(\s+){regex.escape(self.eoc_token)}"
        self.CLEANUP_REPLACE = rf"{self.eoc_token}\1"
        self.CAPITAL_REGEX = r"\s?\b\p{Lu}\p{Ll}+\b"
        self.UPPER_REGEX = r"\s?\b\p{Lu}{2,}\b"
        self.UP_FIND = rf"{regex.escape(self.up_token)}(\s?\w+)"
        self.CAP_FIND = rf"{regex.escape(self.cap_token)}(\s?\w+)"
        self.CYR_FIND = rf"{regex.escape(self.boc_token)}(.*?){regex.escape(self.eoc_token)}"

    def prepare_for_tokenization(self, text: str, is_split_into_words=False, **kwargs):
        def mark_cap(match):
            return f"{self.cap_token}{match.group(0).lower()}"

        def mark_up(match):
             return f"{self.up_token}{match.group(0).lower()}"

        def wrap_and_transliterate(match):
            cyr_text = match.group(0)
            latin_text = cyr2lat(cyr_text)
            return f"{self.boc_token}{latin_text}{self.eoc_token}"

        text = regex.sub(self.CYRILLIC_REGEX, wrap_and_transliterate, text)
        text = regex.sub(self.CLEANUP_FIND, self.CLEANUP_REPLACE, text)
        text = regex.sub(self.CAPITAL_REGEX, mark_cap, text)
        text = regex.sub(self.UPPER_REGEX, mark_up, text)
        return text
