r"""Pure-Python CLIP byte-level BPE tokenizer (OpenAI CLIP `simple_tokenizer`, Hugging Face `CLIPTokenizer`).

Reads the Hugging Face `vocab.json` + `merges.txt` of a checkpoint. The text is cleaned (double `html.unescape`),
whitespace collapsed, lower-cased and split with CLIP's pattern
`<|startoftext|>|<|endoftext|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+`. The stdlib `re` has no
`\p{..}` classes, so letters are `[^\W\d_]` and numbers `\d`: identical for letters and decimal digits; other
numeric characters (superscripts, fractions such as `²`, `½`) are word characters, so they group with letters here
(upstream: one number token each). A literal `<|startoftext|>` / `<|endoftext|>` (exact case) maps to the special
token, as in the Hugging Face `CLIPTokenizer` (the text between them is cleaned separately). ftfy (mojibake repair)
is left out: prompts are expected to be clean Unicode.
"""

from __future__ import annotations

import functools
import html
import itertools
import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_SPECIAL = re.compile(r"(<\|startoftext\|>|<\|endoftext\|>)")
_PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|(?:[^\s\w]|_)+""", re.IGNORECASE)
_NO_MERGE = 1 << 30  # fixed: rank of a pair without a merge rule (above every real rank)


@functools.cache
def _byte_chars() -> dict[int, str]:
    """GPT-2 / CLIP `bytes_to_unicode`: every byte as a printable character."""
    bs = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs), strict=True))


class ClipTokenizer:
    """`text -> [sot, ..., eot]` token ids (unpadded, truncated to the context length)."""

    def __init__(self, vocab: Path, merges: Path, context: int = 32) -> None:
        """Read the vocabulary and the merge rules (first line of `merges.txt`: a version header)."""
        self.encoder: dict[str, int] = json.loads(vocab.read_text(encoding="utf-8"))
        lines = merges.read_text(encoding="utf-8").split("\n")[1:]
        pairs = [tuple(m.split()) for m in lines if m and not m.startswith("#")]
        self.ranks = {(p[0], p[1]): i for i, p in enumerate(pairs)}
        self.sot, self.eot = self.encoder["<|startoftext|>"], self.encoder["<|endoftext|>"]
        self.context = context
        self._cache: dict[str, list[str]] = {}

    def _bpe(self, token: str) -> list[str]:
        if token in self._cache:
            return self._cache[token]
        word = [*token[:-1], token[-1] + "</w>"]
        while len(word) > 1:
            best = min(itertools.pairwise(word), key=lambda p: self.ranks.get(p, _NO_MERGE))
            if best not in self.ranks:
                break
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    out.append(word[i] + word[i + 1])
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = out
        self._cache[token] = word
        return word

    def encode(self, text: str) -> list[int]:
        """Token ids of `text` without the start / end tokens (special tokens in `text` map to their ids)."""
        ids: list[int] = []
        for i, part in enumerate(_SPECIAL.split(text)):
            ids += [self.encoder[part]] if i % 2 else self._encode(part)
        return ids

    def _encode(self, text: str) -> list[int]:
        text = re.sub(r"\s+", " ", html.unescape(html.unescape(text)).strip()).strip().lower()
        bc = _byte_chars()
        ids: list[int] = []
        for tok in _PAT.findall(text):
            ids += [self.encoder[t] for t in self._bpe("".join(bc[b] for b in tok.encode("utf-8")))]
        return ids

    def __call__(self, text: str) -> list[int]:
        """`[sot, ..., eot]`, truncated to the context (the last token stays `eot`), unpadded."""
        ids = [self.sot, *self.encode(text), self.eot]
        if len(ids) > self.context:
            ids = [*ids[: self.context - 1], self.eot]
        return ids
