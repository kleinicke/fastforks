"""Pure-Python BERT WordPiece tokenizer, uncased (google-research/bert `tokenization.py`, Hugging Face `BertTokenizer`).

Reads a checkpoint's `vocab.txt` (one token per line, id = line number). Basic tokenization as upstream: drop control
characters and U+0000 / U+FFFD, whitespace -> space, CJK ideographs split into single characters, lower-case, strip
accents (NFD, drop combining marks), split on whitespace and punctuation (ASCII symbols count as punctuation). Each word
is then split greedily into the longest vocabulary pieces (`##` continuation prefix); a word longer than 100 characters
or without a full split becomes `[UNK]`.
"""

from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_MAX_WORD = 100  # fixed: upstream `max_input_chars_per_word`


def _punct(ch: str) -> bool:
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:  # noqa: PLR2004 - ASCII symbol blocks
        return True
    return unicodedata.category(ch).startswith("P")


def _cjk(cp: int) -> bool:
    return any(
        a <= cp <= b
        for a, b in (
            (0x4E00, 0x9FFF),
            (0x3400, 0x4DBF),
            (0x20000, 0x2A6DF),
            (0x2A700, 0x2B73F),
            (0x2B740, 0x2B81F),
            (0x2B820, 0x2CEAF),
            (0xF900, 0xFAFF),
            (0x2F800, 0x2FA1F),
        )
    )


def _clean(text: str) -> str:
    out = []
    for ch in text:
        cp = ord(ch)
        if cp in (0, 0xFFFD):
            continue
        if ch in "\t\n\r" or unicodedata.category(ch) == "Zs":
            out.append(" ")
        elif unicodedata.category(ch).startswith("C"):
            continue
        elif _cjk(cp):
            out.append(f" {ch} ")
        else:
            out.append(ch)
    return "".join(out)


def basic(text: str) -> list[str]:
    """Words and punctuation marks of `text`, lower-cased, accents stripped."""
    words = []
    for word in _clean(text).split():
        w = "".join(c for c in unicodedata.normalize("NFD", word.lower()) if unicodedata.category(c) != "Mn")
        cur = ""
        for ch in w:
            if _punct(ch):
                if cur:
                    words.append(cur)
                words.append(ch)
                cur = ""
            else:
                cur += ch
        if cur:
            words.append(cur)
    return words


class BertTokenizer:
    """`text -> [CLS] pieces [SEP]` ids (no padding, no truncation)."""

    def __init__(self, vocab: Path) -> None:
        """Load `vocab.txt`."""
        tokens = vocab.read_text(encoding="utf-8").split("\n")
        if tokens and not tokens[-1]:
            tokens.pop()
        self.vocab = {t: i for i, t in enumerate(tokens)}
        self.cls, self.sep, self.unk = self.vocab["[CLS]"], self.vocab["[SEP]"], self.vocab["[UNK]"]

    def pieces(self, word: str) -> list[int]:
        """Greedy longest-match WordPiece ids of one basic word."""
        if len(word) > _MAX_WORD:
            return [self.unk]
        out, start = [], 0
        while start < len(word):
            end = len(word)
            while end > start:
                sub = word[start:end] if start == 0 else "##" + word[start:end]
                if sub in self.vocab:
                    out.append(self.vocab[sub])
                    break
                end -= 1
            else:
                return [self.unk]
            start = end
        return out

    def encode(self, text: str) -> list[int]:
        """Piece ids of `text` without special tokens."""
        return [i for w in basic(text) for i in self.pieces(w)]

    def __call__(self, text: str) -> list[int]:
        """`[CLS] + encode(text) + [SEP]`."""
        return [self.cls, *self.encode(text), self.sep]
