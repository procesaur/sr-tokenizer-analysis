#!/usr/bin/env python3
"""
serbian_preprocessor.py — Text ↔ factored-token transform for the Serbian
expanded-vocabulary Qwen pipeline.

Preprocessing  (text → factored string):
  1. Sanitise and extract protected spans (code, HTML, URLs, math) behind PUA
     placeholders so they are never touched by linguistic transforms.
  2. Detect dominant script (Latin / Cyrillic) of the remaining natural text.
  3. Transliterate Cyrillic → Latin  (srtools; bijective for Serbian).
  4. Normalize casing word-by-word, emitting <|CAP|>, <|UPPER|>, or
     <|CASE|> position-list tags before each non-lowercase word.
  5. In Cyrillic-context documents, mark foreign/technical Latin words with
     inline <|LAT|> so they survive the postprocessing round-trip.
  6. Optionally prepend <|CYR|> at the document start for Cyrillic inputs.
  7. Restore protected spans in place.

Postprocessing (factored string → surface text):
  1. Extract NEW protected spans the model generated (HTML in responses, etc.).
  2. Restore casing from <|CAP|> / <|UPPER|> / <|CASE|> tags.
  3. Determine rendering script and transliterate back to Cyrillic if needed,
     respecting inline <|LAT|> markers.
  4. Restore both model-generated and input-echoed protected spans.
  5. Clean up residual whitespace / spacing around punctuation.

Special tokens used (must be added to the tokenizer as special=True):
    <|CAP|>    Title-case next word
    <|UPPER|>  All-caps next word
    <|CASE|>   Arbitrary casing — followed by comma-separated alpha-positions
    <|CYR|>    Cyrillic context (document-level or response-level)
    <|LAT|>    Keep next word in Latin inside a Cyrillic rendering context

Usage::

    from serbian_preprocessor import SerbianPreprocessor

    sp = SerbianPreprocessor()

    # During training data preparation or inference pre-encode:
    factored, spans = sp.preprocess("Београд је главни град.", insert_script_tag=True)
    token_ids = tokenizer.encode(factored)

    # After model generates and you decode:
    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    surface = sp.postprocess(decoded, input_protected_spans=spans)

Standalone functions ``preprocess()`` and ``postprocess()`` are also
exported for cases where stateless usage is preferred.

Requirements:
    pip install srtools   (Serbian Latin ↔ Cyrillic transliteration)

If srtools is unavailable, Cyrillic ↔ Latin transliteration is skipped with
a warning and the raw Cyrillic text is preserved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from custom.tokenization_srna import lat2cyr as _to_cyrillic, cyr2lat as _to_latin


# ---------------------------------------------------------------------------
# Special token strings (must match what is registered in the tokenizer)
# ---------------------------------------------------------------------------
TOK_CAP   = "<|CAP|>"
TOK_UPPER = "<|UPPER|>"
TOK_CASE  = "<|CASE|>"
TOK_CYR   = "<|CYR|>"
TOK_LAT   = "<|LAT|>"

_ALL_SPECIAL = {TOK_CAP, TOK_UPPER, TOK_CASE, TOK_CYR, TOK_LAT}

# Regex shared with tokenizer wrappers: inserts a space after a preprocessor
# tag only when the next character is NOT another tag.  This avoids an
# orphaned Ġ token between consecutive tags while still letting spaced
# vocabulary tokens (e.g. « beograd») fire after a tag.
#   «<|CYR|>beograd»        → «<|CYR|> beograd»
#   «<|CYR|><|CAP|>beograd» → «<|CYR|><|CAP|> beograd»  (no orphan)
SPACE_INSERT_RE = re.compile(r'(<\|[A-Z]+\|>)(?!<\|)')

# ---------------------------------------------------------------------------
# Protected-span extraction
# PUA code points U+E000 and U+E001 bracket a decimal index.
# These characters are never present in real Serbian/English text.
# ---------------------------------------------------------------------------
_PUA_OPEN  = ""
_PUA_CLOSE = ""
_PLACEHOLDER_RE = re.compile(r"\d+")


def _make_placeholder(idx: int) -> str:
    return f"{_PUA_OPEN}{idx}{_PUA_CLOSE}"


# Protected-span patterns, in priority order (first match wins per position).
# The regex list is applied via re.sub in order; once a span is replaced by a
# PUA placeholder it cannot match any later pattern.
_PROTECTED_PATTERNS: List[re.Pattern] = [
    re.compile(r"```[\s\S]*?```",             re.DOTALL),   # 1 fenced code
    re.compile(r"`[^`\n]+`"),                               # 2 inline code
    re.compile(r"<!--[\s\S]*?-->",            re.DOTALL),   # 3 HTML comments
    re.compile(r"<\?[\s\S]*?\?>",             re.DOTALL),   # 4 XML PI / decl
    re.compile(r"<!\[CDATA\[[\s\S]*?\]\]>",  re.DOTALL),   # 5 CDATA
    re.compile(r"</?[a-zA-Z][^>]*?/?>"),                   # 6 HTML/XML tags
    re.compile(r"\$\$[\s\S]*?\$\$",          re.DOTALL),   # 7 display math
    re.compile(r"\$[^$\n]+?\$"),                            # 8 inline math
    re.compile(r"https?://\S+"),                            # 9 URLs
    re.compile(r"[\w.+-]+@[\w.-]+\.\w+"),                  # 10 emails
    re.compile(r"&(?:[a-zA-Z]+|#\d+|#x[0-9a-fA-F]+);"),  # 11 HTML entities
]


def _extract_protected(text: str) -> Tuple[str, List[str]]:
    """
    Replace protected spans with PUA placeholders (space-padded so the
    placeholder becomes its own whitespace-delimited "word").

    Returns (modified_text, spans) where spans[i] is the original content
    of placeholder index i.
    """
    # Pre-sanitise: strip any pre-existing PUA characters to avoid collision.
    text = text.replace(_PUA_OPEN, "").replace(_PUA_CLOSE, "")

    spans: List[str] = []

    def _replacer(m: re.Match) -> str:
        idx = len(spans)
        spans.append(m.group(0))
        return f" {_make_placeholder(idx)} "

    for pattern in _PROTECTED_PATTERNS:
        text = pattern.sub(_replacer, text)

    return text, spans


def _restore_protected(text: str, spans: List[str]) -> str:
    """Replace PUA placeholders back with their original content."""
    for idx, content in enumerate(spans):
        text = text.replace(_make_placeholder(idx), content)
    # Collapse doubled spaces introduced by space-padding.
    text = re.sub(r" {2,}", " ", text)
    # Remove spurious space before common punctuation.
    text = re.sub(r" ([.,;:!?»\)])", r"\1", text)
    text = re.sub(r"([\(«]) ", r"\1", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Script detection
# ---------------------------------------------------------------------------
_CYR_RE = re.compile(r"[а-яА-ЯёЁђЂљЉњЊћЋџЏјЈ]")
_LAT_ALPHA_RE = re.compile(r"[a-zA-ZčćđšžČĆĐŠŽ]")


def _detect_script(text: str) -> str:
    """Return 'cyrillic' if Cyrillic characters outnumber Latin, else 'latin'."""
    n_cyr = len(_CYR_RE.findall(text))
    n_lat = len(_LAT_ALPHA_RE.findall(text))
    return "cyrillic" if n_cyr > n_lat else "latin"


# ---------------------------------------------------------------------------
# Casing normalisation
# ---------------------------------------------------------------------------

def _split_punctuation(word: str) -> Tuple[str, str, str]:
    """
    Split *word* into (prefix_punct, alpha_core, suffix_punct).

    Leading punctuation is everything before the first alphanumeric character
    (letters OR digits).  Digits that start a word — like the '3' in '3D' —
    are NOT split off; they stay attached to the alpha_core so that the
    whole token is lowercased together and the casing tag precedes it.

    Trailing punctuation is everything after the last *alphabetic* character
    (digits that trail the last alpha, e.g. 'CO2', stay inside the core so
    that the postprocessor reproduces them correctly).

    Internal non-alpha characters (apostrophes, hyphens, embedded digits)
    stay inside alpha_core so that position indexing works correctly for
    O'Brien, Jean-Paul, 3D, CO2, OAuth2, etc.
    """
    if not word:
        return "", "", ""

    # Leading split: first alphanumeric (alpha OR digit) character.
    first_alnum = -1
    for i, ch in enumerate(word):
        if ch.isalpha() or ch.isdigit():
            first_alnum = i
            break

    if first_alnum == -1:
        # Pure punctuation — no usable core.
        return "", "", word

    # Trailing split: last alphabetic character (digits after last alpha are
    # part of the core, e.g. 'CO2' → core='CO2', suffix='').
    last_alpha = -1
    for i, ch in enumerate(word):
        if ch.isalpha():
            last_alpha = i

    if last_alpha == -1:
        # All digits / punctuation — treat as pass-through.
        return "", "", word

    return word[:first_alnum], word[first_alnum: last_alpha + 1], word[last_alpha + 1:]


def _alpha_chars(s: str) -> List[Tuple[int, str]]:
    """Return list of (original_index_in_s, char) for alphabetic characters."""
    return [(i, ch) for i, ch in enumerate(s) if ch.isalpha()]


def _classify_casing(alpha_core: str) -> Tuple[Optional[str], str, str]:
    """
    Analyse the casing pattern of *alpha_core* and return:
        (tag, position_string, lowered_core)

    tag is one of TOK_CAP, TOK_UPPER, TOK_CASE, or None (all lowercase).
    position_string is the comma-separated list of uppercase alpha-char
    positions (used only with TOK_CASE); empty string otherwise.
    lowered_core is alpha_core.lower().
    """
    lowered = alpha_core.lower()
    alphas = _alpha_chars(alpha_core)

    if not alphas:
        return None, "", lowered

    upper_alpha_indices = [ai for ai, (_, ch) in enumerate(alphas) if ch.isupper()]
    n_alpha = len(alphas)
    n_upper = len(upper_alpha_indices)

    if n_upper == 0:
        return None, "", lowered

    if n_alpha == 1:
        # Single alpha character, always use CAP (CAP ≡ UPPER for length 1).
        return TOK_CAP, "", lowered

    if n_upper == n_alpha:
        # All alpha characters are uppercase → UPPER.
        return TOK_UPPER, "", lowered

    if upper_alpha_indices == [0]:
        # Only the first alpha character is uppercase → Title case → CAP.
        return TOK_CAP, "", lowered

    # Arbitrary / mixed-case (CamelCase, pH, mRNA, etc.) → CASE + positions.
    pos_str = ",".join(str(p) for p in upper_alpha_indices)
    return TOK_CASE, pos_str, lowered


def _normalize_casing(text: str) -> str:
    """
    Inject casing tags into *text* word by word.

    Words in the text are whitespace-delimited.  PUA placeholders (from
    protected-span extraction) contain no alphabetic characters and pass
    through untouched.
    """
    # Use a single-space split to preserve empty strings from multiple spaces.
    words = text.split(" ")
    result: List[str] = []

    for word in words:
        if not word:
            result.append("")
            continue

        # PUA placeholders — pass through unchanged.
        if _PLACEHOLDER_RE.fullmatch(word):
            result.append(word)
            continue

        prefix, core, suffix = _split_punctuation(word)

        if not core or not any(ch.isalpha() for ch in core):
            # Pure punctuation, digits, or empty core.
            result.append(word)
            continue

        tag, pos_str, lowered_core = _classify_casing(core)
        lowered_word = prefix + lowered_core + suffix

        if tag is None:
            result.append(lowered_word)
        elif tag == TOK_CAP:
            result.append(f"{prefix}{TOK_CAP}{lowered_core}{suffix}")
        elif tag == TOK_UPPER:
            result.append(f"{prefix}{TOK_UPPER}{lowered_core}{suffix}")
        else:  # TOK_CASE
            result.append(f"{prefix}{TOK_CASE}{pos_str}{lowered_core}{suffix}")

    return " ".join(result)


# ---------------------------------------------------------------------------
# Inline Latin detection (for Cyrillic-context documents)
# ---------------------------------------------------------------------------

# Characters not in the Serbian Latin alphabet.  A word containing any of
# these is almost certainly a foreign term that should stay Latin.
_NON_SR_LATIN = frozenset("wxyqWXYQ")

# Serbian Latin alphabet characters (lowercase).
_SR_LATIN = frozenset("abcčćdđefghijklmnoprsštuvzž")

# Pattern for URLs / emails / file paths / code-ish tokens (after lowercasing).
_TECHNICAL_RE = re.compile(
    r"(?:"
    r"https?://"                        # URL
    r"|www\."                           # www prefix
    r"|[\w.+-]+@[\w.-]+\.\w+"          # email
    r"|[a-z]:[\\/]"                     # Windows path
    r"|/[a-z]"                          # Unix path
    r"|[a-z_]+\.[a-z]{2,4}$"           # file extension
    r"|[a-z_]+\(\)"                     # function call
    r"|[a-z]+_[a-z_]+"                 # snake_case
    r")",
    re.IGNORECASE,
)

# Curated set of common technical / foreign terms that should stay Latin in
# Cyrillic-context text.  Keyed by lowercase form.  Extensible at runtime.
FOREIGN_TERM_DICT: frozenset = frozenset({
    # Programming languages & runtimes
    "python", "javascript", "typescript", "java", "kotlin", "swift",
    "golang", "rust", "ruby", "php", "perl", "scala", "haskell",
    "matlab", "fortran", "cobol", "pascal",
    # Web / protocols
    "html", "css", "http", "https", "ftp", "ssh", "smtp", "dns",
    "json", "xml", "yaml", "sql", "api", "rest", "graphql", "oauth",
    "websocket", "grpc", "cors",
    # Platforms & tools
    "linux", "ubuntu", "debian", "fedora", "windows", "macos", "android",
    "ios", "docker", "kubernetes", "git", "github", "gitlab", "bitbucket",
    "npm", "pip", "conda", "gradle", "maven", "webpack", "vite",
    # Cloud / infra
    "aws", "azure", "gcp", "terraform", "ansible", "nginx", "apache",
    # AI / ML
    "pytorch", "tensorflow", "keras", "sklearn", "pandas", "numpy",
    "jupyter", "cuda", "gpu", "cpu", "ram", "chatgpt", "openai", "llm",
    # Brands / devices
    "google", "facebook", "meta", "twitter", "instagram", "youtube",
    "whatsapp", "telegram", "tiktok", "netflix", "spotify", "amazon",
    "apple", "samsung", "huawei", "microsoft", "intel", "nvidia",
    "iphone", "ipad", "android", "playstation", "xbox",
    # Tech terms
    "wifi", "bluetooth", "usb", "hdmi", "pdf", "bitcoin", "blockchain",
    "nft", "vpn", "tor", "proxy",
})


def _detect_inline_latin(word_lower: str) -> bool:
    """Return True if *word_lower* (already lowercased) should stay Latin."""
    if any(ch in _NON_SR_LATIN for ch in word_lower):
        return True
    if word_lower in FOREIGN_TERM_DICT:
        return True
    if _TECHNICAL_RE.search(word_lower):
        return True
    return False


def _insert_lat_markers(text: str) -> str:
    """
    Scan words in *text* (already case-normalised and transliterated to Latin).
    Prepend <|LAT|> before any word that should stay Latin in a
    Cyrillic-context document.

    Only called when the source document was Cyrillic.

    Handles both space-separated tags (legacy) and space-free merged tags
    (current factored format where <|CAP|>radnik is a single token after
    whitespace split).
    """
    tokens = text.split(" ")
    result: List[str] = []

    for tok in tokens:
        if not tok:
            result.append("")
            continue

        # PUA placeholders — pass through unchanged.
        if _PLACEHOLDER_RE.fullmatch(tok):
            result.append(tok)
            continue

        # Split the token into (casing_tags, word_core).
        # Space-free format: "<|CAP|>radnik" or "<|CASE|>1,2,3mrna"
        # Space-separated format: casing tag and word are separate tokens
        casing_prefix = ""
        word_remainder = tok

        if word_remainder.startswith(TOK_CAP):
            casing_prefix = TOK_CAP
            word_remainder = word_remainder[len(TOK_CAP):]
        elif word_remainder.startswith(TOK_UPPER):
            casing_prefix = TOK_UPPER
            word_remainder = word_remainder[len(TOK_UPPER):]
        elif word_remainder.startswith(TOK_CASE):
            casing_prefix = TOK_CASE
            after_case = word_remainder[len(TOK_CASE):]
            # Read the position digits (or empty if none).
            pos_end = 0
            while pos_end < len(after_case) and after_case[pos_end] in "0123456789,":
                pos_end += 1
            if pos_end > 0:
                casing_prefix += after_case[:pos_end]
            word_remainder = after_case[pos_end:]

        if word_remainder:
            _, core, _ = _split_punctuation(word_remainder)
            core_lower = core.lower()
            if core_lower and _detect_inline_latin(core_lower):
                result.append(casing_prefix + TOK_LAT + word_remainder)
            else:
                result.append(tok)  # keep original
        elif casing_prefix:
            # Pure casing tag with no word (legacy space-separated format).
            result.append(casing_prefix)
        else:
            result.append(tok)

    return " ".join(result)


# ---------------------------------------------------------------------------
# Main preprocessing entry point
# ---------------------------------------------------------------------------

def preprocess(
    text: str,
    insert_script_tag: bool = False,
    factor_casing: bool = True,
) -> Tuple[str, List[str]]:
    """
    Transform *text* into its factored representation.

    Parameters
    ----------
    text:
        Raw surface text (may be Latin or Cyrillic, any casing).
    insert_script_tag:
        If True and the document is Cyrillic, prepend <|CYR|>.
        Set True when preparing training data and at inference time.
        Set False when processing plain pretraining text (CPT) if you
        prefer not to clutter every block with the tag.
    factor_casing:
        If True (default), emit <|CAP|>/<|UPPER|>/<|CASE|> tags and
        lowercase all words.  If False, preserve surface casing — use
        with surface-casing vocabulary mode where cased variants are
        directly in the tokenizer (no casing tags needed).

    Returns
    -------
    factored_text:
        The normalized string ready to pass to ``tokenizer.encode()``.
    spans:
        List of protected-span contents (original HTML tags, code, URLs …).
        Pass this back to ``postprocess()`` as ``input_protected_spans``.
    """
    # Step 0: extract protected spans.
    text, spans = _extract_protected(text)

    # Step 1: detect script on non-protected remainder.
    script = _detect_script(text)

    # Step 2: transliterate Cyrillic → Latin.
    if script == "cyrillic":
        text = _to_latin(text)

    # Step 3: normalize casing (skip if surface-casing vocabulary).
    if factor_casing:
        text = _normalize_casing(text)

    # Step 4: insert inline <|LAT|> markers (Cyrillic context only).
    if script == "cyrillic":
        text = _insert_lat_markers(text)

    # Step 5: prepend document-level <|CYR|>.
    # No space after CYR — the word immediately following is naturally
    # separated from the special token by the ByteLevel pre-tokenizer;
    # a literal space would add an orphaned Ġ token.
    if insert_script_tag and script == "cyrillic":
        text = f"{TOK_CYR}{text}"

    # Step 6: restore protected spans.
    text = _restore_protected(text, spans)

    return text, spans


# ---------------------------------------------------------------------------
# Casing restoration (postprocessing step 1)
# ---------------------------------------------------------------------------

# CamelCase fallback dictionary.  Keys: lowercase form.  Values: canonical
# surface form.  Used only when the model emits a malformed <|CASE|> payload.
CAMELCASE_DICT: dict[str, str] = {
    "mcdonald":      "McDonald",
    "mcdonald's":    "McDonald's",
    "iphone":        "iPhone",
    "ipad":          "iPad",
    "youtube":       "YouTube",
    "javascript":    "JavaScript",
    "typescript":    "TypeScript",
    "pytorch":       "PyTorch",
    "tensorflow":    "TensorFlow",
    "macos":         "macOS",
    "ph":            "pH",
    "mrna":          "mRNA",
    "dsdna":         "dsDNA",
    "latex":         "LaTeX",
    "linkedin":      "LinkedIn",
    "tiktok":        "TikTok",
    "github":        "GitHub",
    "gitlab":        "GitLab",
    "openai":        "OpenAI",
    "chatgpt":       "ChatGPT",
    "playstation":   "PlayStation",
    "wordpress":     "WordPress",
    "whatsapp":      "WhatsApp",
    "instagram":     "Instagram",
}


def _skip_space(text: str, i: int) -> int:
    """Advance past exactly one space character if present."""
    if i < len(text) and text[i] == " ":
        return i + 1
    return i


def _at_word_boundary(text: str, i: int) -> bool:
    """True if position i is at a space or a special-token boundary."""
    if i >= len(text):
        return True
    if text[i] == " ":
        return True
    for tok in _ALL_SPECIAL:
        if text[i:].startswith(tok):
            return True
    return False


def _parse_positions(pos_str: str) -> Optional[frozenset]:
    """Parse comma-separated integer positions; return None on failure."""
    if not pos_str:
        return None
    try:
        return frozenset(int(p) for p in pos_str.split(",") if p)
    except ValueError:
        return None


def _capitalize_first_alpha(s: str) -> str:
    """Upper-case the first alphabetic character in *s*, leave rest as-is."""
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1:]
    return s


def _restore_casing(text: str) -> str:
    """
    Walk *text* left-to-right and apply casing tags to the words that follow
    them.  Tags themselves are consumed (not emitted into output).

    Consecutive tags: the last tag before a word wins.
    Truncated tag at end-of-string: silently discarded.
    """
    result: List[str] = []
    i = 0
    n = len(text)

    while i < n:
        # ------------------------------------------------------------------ CAP
        if text[i:].startswith(TOK_CAP):
            i += len(TOK_CAP)
            i = _skip_space(text, i)
            # Pass through an inline <|LAT|> that sits between the casing tag
            # and the word (e.g. <|CAP|><|LAT|>python → Python kept Latin).
            if text[i:].startswith(TOK_LAT):
                result.append(TOK_LAT)
                i += len(TOK_LAT)
                i = _skip_space(text, i)
            # Copy any leading non-alpha punctuation (brackets, quotes …).
            while i < n and not text[i].isalpha() and not _at_word_boundary(text, i):
                result.append(text[i])
                i += 1
            # Capitalize first alpha char.
            if i < n and text[i].isalpha():
                result.append(text[i].upper())
                i += 1
            # Copy rest of word verbatim until next boundary.
            while i < n and not _at_word_boundary(text, i):
                result.append(text[i])
                i += 1

        # ---------------------------------------------------------------- UPPER
        elif text[i:].startswith(TOK_UPPER):
            i += len(TOK_UPPER)
            i = _skip_space(text, i)
            # Same <|LAT|> pass-through as for CAP.
            if text[i:].startswith(TOK_LAT):
                result.append(TOK_LAT)
                i += len(TOK_LAT)
                i = _skip_space(text, i)
            while i < n and not text[i].isalpha() and not _at_word_boundary(text, i):
                result.append(text[i])
                i += 1
            # Uppercase everything until word boundary.
            while i < n and not _at_word_boundary(text, i):
                result.append(text[i].upper())
                i += 1

        # ----------------------------------------------------------------- CASE
        elif text[i:].startswith(TOK_CASE):
            i += len(TOK_CASE)
            i = _skip_space(text, i)

            # Read position string (digits and commas only).
            pos_start = i
            while i < n and text[i] in "0123456789,":
                i += 1
            pos_str = text[pos_start:i]
            i = _skip_space(text, i)

            # Positions set (may be None if malformed).
            positions = _parse_positions(pos_str)

            # Check for inline <|LAT|> between positions and word — pass it
            # through for the script-restoration step, but don't let it
            # interfere with word reading here.
            if text[i:].startswith(TOK_LAT):
                result.append(TOK_LAT)
                i += len(TOK_LAT)
                i = _skip_space(text, i)

            # Copy any leading punctuation.
            while i < n and not text[i].isalpha() and not _at_word_boundary(text, i):
                result.append(text[i])
                i += 1

            # Collect the word, applying uppercase at specified alpha-positions.
            word_buf: List[str] = []
            alpha_idx = 0
            word_start_in_result = len(result)  # for dictionary fallback

            while i < n and not _at_word_boundary(text, i):
                ch = text[i]
                if ch.isalpha():
                    if positions is not None and alpha_idx in positions:
                        word_buf.append(ch.upper())
                    else:
                        word_buf.append(ch)
                    alpha_idx += 1
                else:
                    word_buf.append(ch)
                i += 1

            word_str = "".join(word_buf)

            if positions is None:
                # Malformed positions — try dictionary fallback.
                word_lower = word_str.lower()
                if word_lower in CAMELCASE_DICT:
                    word_str = CAMELCASE_DICT[word_lower]
                else:
                    word_str = _capitalize_first_alpha(word_str)

            result.append(word_str)

        # Strip stray / document-level script tags that reach the casing
        # restorer (they are consumed by determine_script before this, but
        # be defensive).
        elif text[i:].startswith(TOK_CYR):
            result.append(TOK_CYR)  # pass through for script restorer
            i += len(TOK_CYR)

        elif text[i:].startswith(TOK_LAT):
            result.append(TOK_LAT)  # pass through for script restorer
            i += len(TOK_LAT)

        else:
            result.append(text[i])
            i += 1

    return "".join(result)


# ---------------------------------------------------------------------------
# Script restoration (postprocessing step 2)
# ---------------------------------------------------------------------------

def _determine_script(
    response_text: str,
    context_has_cyr: bool,
    api_override: Optional[str] = None,
) -> str:
    """
    Determine the rendering script for *response_text* using the precedence
    hierarchy defined in section 4.6 of the design document:

      1. Model-emitted document-level tag (FIRST special token of the response)
      2. API override parameter
      3. Context signal (did the input carry <|CYR|>?)
      4. Default: latin

    Only the FIRST special token at the start of the response is treated as a
    document-level script selector.  Any <|LAT|> or <|CYR|> that appears
    later — including immediately after a doc-level <|CYR|> tag — is treated
    as an inline marker by _restore_script, not as a second document-level
    override.  This correctly handles the common pattern
    "<|CYR|> <|CAP|> <|LAT|> Python ..." where <|LAT|> is inline.
    """
    stripped = response_text.lstrip()

    # Priority 1: look at the very first special token only.
    if stripped.startswith(TOK_CYR):
        return "cyrillic"
    if stripped.startswith(TOK_LAT):
        return "latin"

    # Priority 2: API override.
    if api_override in ("cyrillic", "latin"):
        return api_override

    # Priority 3: context.
    if context_has_cyr:
        return "cyrillic"

    # Priority 4: default.
    return "latin"


def _strip_leading_doc_tags(text: str) -> str:
    """
    Remove exactly ONE document-level script tag from the very start of
    *text*.  Only the first tag is document-level; any subsequent <|LAT|> or
    <|CYR|> belongs to the inline stream and must be left for _restore_script
    to handle.
    """
    stripped = text.lstrip()
    for tag in (TOK_CYR, TOK_LAT):
        if stripped.startswith(tag):
            stripped = stripped[len(tag):]
            stripped = stripped.lstrip()
            break  # stop after the first match — don't eat inline markers
    return stripped


@dataclass
class _Segment:
    text: str
    is_protected: bool  # True → keep Latin; False → may transliterate


def _split_on_lat_markers(text: str) -> List[_Segment]:
    """
    Split *text* into alternating unprotected / LAT-protected segments.

    Each <|LAT|> marker protects the immediately following word (until the
    next whitespace).
    """
    segments: List[_Segment] = []
    i = 0
    n = len(text)
    buf: List[str] = []

    while i < n:
        if text[i:].startswith(TOK_LAT):
            # Flush unprotected buffer.
            if buf:
                segments.append(_Segment("".join(buf), is_protected=False))
                buf = []

            i += len(TOK_LAT)
            # Skip optional space after marker.
            if i < n and text[i] == " ":
                i += 1

            # Capture the protected word (until next whitespace).
            word_buf: List[str] = []
            while i < n and text[i] != " ":
                word_buf.append(text[i])
                i += 1
            segments.append(_Segment("".join(word_buf), is_protected=True))

            # Carry the trailing space into the unprotected buffer.
            if i < n and text[i] == " ":
                buf.append(" ")
                i += 1
        else:
            buf.append(text[i])
            i += 1

    if buf:
        segments.append(_Segment("".join(buf), is_protected=False))

    return segments


def _restore_script(text: str, rendering_script: str) -> str:
    """
    Apply script rendering to *text* (which has already been case-restored).

    In Latin mode: strip stray <|LAT|> and <|CYR|> tags, no transliteration.
    In Cyrillic mode: transliterate unprotected segments; keep <|LAT|>-marked
    words in Latin.
    """
    # Strip document-level tag from the start (already consumed by
    # _determine_script, but be defensive).
    text = _strip_leading_doc_tags(text)

    if rendering_script == "latin":
        text = text.replace(f"{TOK_LAT} ", "").replace(TOK_LAT, "")
        text = text.replace(f"{TOK_CYR} ", "").replace(TOK_CYR, "")
        return text

    # Cyrillic mode.
    # Strip any stray mid-text <|CYR|> tags (malformed — not supported mid-stream).
    text = text.replace(f"{TOK_CYR} ", "").replace(TOK_CYR, "")

    segments = _split_on_lat_markers(text)
    parts: List[str] = []
    for seg in segments:
        if seg.is_protected:
            parts.append(seg.text)
        else:
            parts.append(_to_cyrillic(seg.text))

    return "".join(parts)


# ---------------------------------------------------------------------------
# Main postprocessing entry point
# ---------------------------------------------------------------------------

def postprocess(
    text: str,
    context_has_cyr: bool = False,
    api_override: Optional[str] = None,
    input_protected_spans: Optional[List[str]] = None,
) -> str:
    """
    Transform a decoded (``skip_special_tokens=False``) model output back to
    surface text.

    Parameters
    ----------
    text:
        Output of ``tokenizer.decode(ids, skip_special_tokens=False)``.
    context_has_cyr:
        True if the input prompt contained / was tagged with <|CYR|>.
        Controls the fallback script when the model doesn't emit a doc-level tag.
    api_override:
        If ``"cyrillic"`` or ``"latin"``, forces that script (priority 2).
    input_protected_spans:
        The ``spans`` list returned by ``preprocess()`` for the corresponding
        input.  Used to restore structural content the model echoed verbatim.
        Pass ``None`` if not available (spans won't be restored from input).

    Returns
    -------
    surface_text:
        Fully restored surface text.
    """
    # Step 0a: extract NEW protected spans the model generated (HTML in
    # responses, code blocks it wrote, etc.).
    text, output_spans = _extract_protected(text)

    # Determine rendering script BEFORE removing doc-level tags from text.
    rendering_script = _determine_script(text, context_has_cyr, api_override)

    # Step 1: restore casing.
    text = _restore_casing(text)

    # Step 2: restore script.
    text = _restore_script(text, rendering_script)

    # Step 3a: restore model-generated protected spans.
    text = _restore_protected(text, output_spans)

    # Step 3b: restore input-echoed placeholders (if any survived in output).
    if input_protected_spans:
        text = _restore_protected(text, input_protected_spans)

    # Step 4: final cleanup.
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r" ([.,;:!?»\)])", r"\1", text)
    text = re.sub(r"([\(«]) ", r"\1", text)
    text = text.strip()

    return text


# ---------------------------------------------------------------------------
# Stateful wrapper (recommended for inference)
# ---------------------------------------------------------------------------

class SerbianPreprocessor:
    """
    Stateful wrapper that holds protected spans across a request/response
    cycle so that postprocess() can restore structural content echoed from
    the input.

    Typical usage (inference)::

        sp = SerbianPreprocessor()

        # Pre-encode user message:
        factored = sp.encode("Пример текста.")
        ids = tokenizer.encode(factored)

        # Decode model output:
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        surface = sp.decode(decoded, api_override="cyrillic")

    For training data preparation, use the module-level ``preprocess()``
    and ``postprocess()`` functions directly (no state needed between calls).
    """

    def __init__(
        self,
        insert_script_tag: bool = True,
        factor_casing: bool = True,
    ) -> None:
        self._insert_script_tag = insert_script_tag
        self._factor_casing = factor_casing
        self._last_spans: List[str] = []
        self._last_context_has_cyr: bool = False

    def encode(self, text: str) -> str:
        """
        Preprocess *text* and store protected spans + script context for the
        subsequent ``decode()`` call.  Returns the factored string.
        """
        factored, spans = preprocess(
            text,
            insert_script_tag=self._insert_script_tag,
            factor_casing=self._factor_casing,
        )
        self._last_spans = spans
        self._last_context_has_cyr = TOK_CYR in factored and factored.startswith(TOK_CYR)
        return factored

    def decode(
        self,
        model_output: str,
        api_override: Optional[str] = None,
    ) -> str:
        """
        Postprocess a decoded model output using state from the last
        ``encode()`` call.
        """
        return postprocess(
            model_output,
            context_has_cyr=self._last_context_has_cyr,
            api_override=api_override,
            input_protected_spans=self._last_spans,
        )

    def reset(self) -> None:
        """Clear stored state (call between independent requests if reusing)."""
        self._last_spans = []
        self._last_context_has_cyr = False


# ---------------------------------------------------------------------------
# Self-test / smoke-check
# ---------------------------------------------------------------------------

def _run_tests() -> None:
    """Quick smoke-check of the preprocessing round-trip."""
    import sys, traceback
    # Force UTF-8 output on Windows consoles that default to cp1252.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    CASES = [
        # (raw_input, insert_script_tag, expected_fragments_in_factored)
        # Basic Title case
        ("Radnik radi.", False, [f"{TOK_CAP}radnik"]),
        # UPPER
        ("NATO savez.", False, [f"{TOK_UPPER}nato"]),
        # CamelCase
        ("Koristim iPhone.", False, [f"{TOK_CASE}1iphone"]),
        # pH (single CASE)
        ("vrednost pH vode.", False, [f"{TOK_CASE}1ph"]),
        # mRNA
        ("mRNA vakcina.", False, [f"{TOK_CASE}1,2,3mrna"]),
        # PyTorch
        ("koristim PyTorch.", False, [f"{TOK_CASE}0,2pytorch"]),
        # McDonald's
        ("bio sam u McDonald's.", False, [f"{TOK_CASE}0,2mcdonald's"]),
        # Jean-Paul
        ("čitam Jean-Paul Sartre.", False, [f"{TOK_CASE}0,4jean-paul"]),
        # 3D (single alpha → CAP)
        ("štampač za 3D modele.", False, [f"{TOK_CAP}3d"]),
        # macOS
        ("sistem macOS.", False, [f"{TOK_CASE}3,4macos"]),
        # Cyrillic input → Latin + CYR tag (no space after CYR)
        ("Радник ради.", True, [TOK_CYR + TOK_CAP + "radnik"]),
        # Protected span (HTML) preserved
        ("<b>radnik</b>", False, ["<b>", "</b>", "radnik"]),
        # O'Brien
        ("susreo sam O'Brien.", False, [f"{TOK_CASE}0,1o'brien"]),
        # All lower — no tag
        ("radnik radi.", False, ["radnik radi."]),
        # UPPER single char — CAP
        ("U Beogradu.", False, [f"{TOK_CAP}u"]),
        # LaTeX
        ("pišem u LaTeX.", False, [f"{TOK_CASE}0,2,4latex"]),
    ]

    passed = 0
    failed = 0
    for raw, script_tag, fragments in CASES:
        try:
            factored, _ = preprocess(raw, insert_script_tag=script_tag)
            ok = all(frag in factored for frag in fragments)
            status = "PASS" if ok else "FAIL"
            if not ok:
                failed += 1
                print(f"  {status}  input={raw!r}")
                print(f"        factored={factored!r}")
                print(f"        expected fragments={fragments}")
            else:
                passed += 1
                print(f"  {status}  {raw!r}")
        except Exception:
            failed += 1
            print(f"  ERROR input={raw!r}")
            traceback.print_exc()

    print(f"\n{passed} passed, {failed} failed.")

    # Postprocessing round-trip
    print("\n--- Postprocessing round-trip ---")
    POST_CASES = [
        # (factored, has_cyr, expected_surface)
        ("<|CYR|> <|CAP|> beograd je prestonica <|CAP|> srbije. <|UPPER|> nato ima mnogo članica. <|CAP|> <|LAT|> python je jezik.",
         True, "Београд је престоница Србије. НАТО има много чланица. Python је језик."),
        ("<|CAP|> radnik radi.",                         False, "Radnik radi."),
        ("<|CASE|> 1 iphone je dobar.",                  False, "iPhone je dobar."),
        ("<|CASE|> 0,2 mcdonald's je brza hrana.",       False, "McDonald's je brza hrana."),
        ("<|CASE|> 1,2,3 mrna vakcina.",                 False, "mRNA vakcina."),
        ("<|UPPER|> nato <|CAP|> savez.",                False, "NATO Savez."),
        ("<|CASE|> 0,4 jean-paul <|CAP|> sartre.",       False, "Jean-Paul Sartre."),
        # CAP + LAT in Cyrillic context → capitalized word stays Latin
        ("<|CYR|> <|CAP|> <|LAT|> python je dobar.",    True,  "Python је добар."),
        # UPPER + LAT in Cyrillic context → all-caps word stays Latin
        ("<|CYR|> <|UPPER|> <|LAT|> api je brz.",       True,  "API је брз."),
        # CASE + LAT in Cyrillic context
        ("<|CYR|> <|CASE|> 0,2 <|LAT|> pytorch je biblioteka.", True, "PyTorch је библиотека."),
        # CAP at start of string (no preceding space)
        ("<|CAP|> zemlja.",                              False, "Zemlja."),
        # UPPER single char → CAP rule (single alpha → CAP in preprocessing)
        ("<|CAP|> u beogradu.",                          False, "U beogradu."),
        # Bracket punctuation before word
        ("(<|UPPER|> nato)",                             False, "(NATO)"),
        # Quote punctuation
        ('<|CAP|> "radnik"',                             False, '"Radnik"'),
        # --- Space-free format (current preprocessor output) ---
        ("<|CYR|><|CAP|>beograd je prestonica <|CAP|>srbije. <|UPPER|>nato ima mnogo članica. <|CAP|><|LAT|>python je jezik.",
         True, "Београд је престоница Србије. НАТО има много чланица. Python је језик."),
        ("<|CAP|>radnik radi.",                             False, "Radnik radi."),
        ("<|CASE|>1iphone je dobar.",                       False, "iPhone je dobar."),
        ("<|CASE|>0,2mcdonald's je brza hrana.",            False, "McDonald's je brza hrana."),
        ("<|CASE|>1,2,3mrna vakcina.",                      False, "mRNA vakcina."),
        ("<|UPPER|>nato <|CAP|>savez.",                     False, "NATO Savez."),
        ("<|CYR|><|CAP|><|LAT|>python je dobar.",          True,  "Python је добар."),
        ("<|CYR|><|UPPER|><|LAT|>api je brz.",             True,  "API је брз."),
        ("<|CYR|><|CASE|>0,2<|LAT|>pytorch je biblioteka.", True, "PyTorch је библиотека."),
    ]

    post_pass = 0
    post_fail = 0
    for factored, has_cyr, expected in POST_CASES:
        try:
            surface = postprocess(factored, context_has_cyr=has_cyr)
            ok = surface == expected
            status = "PASS" if ok else "FAIL"
            if ok:
                post_pass += 1
                print(f"  {status}  {factored!r}")
            else:
                post_fail += 1
                print(f"  {status}  {factored!r}")
                print(f"        got      {surface!r}")
                print(f"        expected {expected!r}")
        except Exception:
            post_fail += 1
            print(f"  ERROR: {factored!r}")
            traceback.print_exc()

    print(f"\n{post_pass} passed, {post_fail} failed.")


def _insert_spaces(text):
    return SPACE_INSERT_RE.sub(r'\1 ', text)

def preprocess1(text):
    text, _ = preprocess(text, insert_script_tag=True, factor_casing=False)
    return _insert_spaces(text)

def preprocess2(text):
    text, _ =preprocess(text, insert_script_tag=True, factor_casing=True)
    return _insert_spaces(text)


if __name__ == "__main__":
    _run_tests()
