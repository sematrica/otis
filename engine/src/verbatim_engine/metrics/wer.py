"""Word and character error rates against a reference transcript.

Normalisation is deliberately simple and documented so results are
reproducible: Unicode NFKC, case-folding, punctuation removed (apostrophes
inside words kept), hyphens and whitespace collapsed to single spaces.
Numbers are *not* normalised ("ten" ≠ "10").
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("’", "'")
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch == "'":
            out.append(ch)
        elif cat.startswith("P") or cat.startswith("S"):
            out.append(" ")
        else:
            out.append(ch)
    text = "".join(out)
    text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)  # quotes, not apostrophes
    return " ".join(text.split())


def _edit_distance(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    """Returns (distance, substitutions, deletions, insertions)."""
    rows, cols = len(ref) + 1, len(hyp) + 1
    # dp[i][j] = (cost, subs, dels, ins)
    prev = [(j, 0, 0, j) for j in range(cols)]
    for i in range(1, rows):
        cur = [(i, 0, i, 0)] + [(0, 0, 0, 0)] * (cols - 1)
        for j in range(1, cols):
            if ref[i - 1] == hyp[j - 1]:
                cur[j] = prev[j - 1]
                continue
            sub = prev[j - 1]
            dele = prev[j]
            ins = cur[j - 1]
            best = min(sub[0], dele[0], ins[0])
            if best == sub[0]:
                cur[j] = (sub[0] + 1, sub[1] + 1, sub[2], sub[3])
            elif best == dele[0]:
                cur[j] = (dele[0] + 1, dele[1], dele[2] + 1, dele[3])
            else:
                cur[j] = (ins[0] + 1, ins[1], ins[2], ins[3] + 1)
        prev = cur
    return prev[-1]


@dataclass(frozen=True)
class ErrorRate:
    rate: float
    errors: int
    reference_length: int
    substitutions: int
    deletions: int
    insertions: int


def word_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    dist, s, d, i = _edit_distance(ref, hyp)
    return ErrorRate(dist / max(1, len(ref)), dist, len(ref), s, d, i)


def character_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    ref, hyp = list(normalize(reference).replace(" ", "")), list(normalize(hypothesis).replace(" ", ""))
    dist, s, d, i = _edit_distance(ref, hyp)
    return ErrorRate(dist / max(1, len(ref)), dist, len(ref), s, d, i)
