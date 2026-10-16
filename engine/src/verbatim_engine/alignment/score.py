"""Acoustic evidence for alternative wordings of a passage.

A CTC acoustic model (the word-alignment model) gives, for every audio frame,
log-probabilities of its tokens. Summed over every way of emitting a text in
order (the CTC forward algorithm), they give ``log P(text | audio)``: how well
the sounds in the audio support that wording. The model has no language
model of the recogniser's kind, so this is evidence independent of Whisper -
the basis on which Maximum Accuracy mode chooses between transcription
candidates ("prefer wording supported by acoustic evidence").

Scores are only comparable between texts scored on the *same* emissions
(the same audio window). The empty text is a valid hypothesis too: its score
says how well "nobody said anything here" explains the audio.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..audio.level import even_out
from .ctc import min_frames
from .models import CTCModel, NemoCTCModel
from .numbers import readings

#: At most this many combinations of number readings are scored per text.
MAX_COMBINATIONS = 16


def ctc_log_likelihood(log_probs: np.ndarray, targets: Sequence[int], blank: int) -> float | None:
    """``log P(targets | frames)`` summed over all CTC paths (forward algorithm).

    ``None`` if the audio is too short to emit the targets. With no targets,
    every frame must be blank.
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    frames = lp.shape[0]
    if frames == 0:
        return None
    if not targets:
        return float(lp[:, blank].sum())
    if blank in targets:
        raise ValueError("targets must not contain the blank token")
    if frames < min_frames(list(targets)):
        return None
    ext = np.full(2 * len(targets) + 1, blank, dtype=np.int64)
    ext[1::2] = targets
    states = ext.size
    can_skip = np.zeros(states, dtype=bool)
    can_skip[3::2] = ext[3::2] != ext[1:-2:2]

    alpha = np.full(states, -np.inf)
    alpha[0] = lp[0, ext[0]]
    alpha[1] = lp[0, ext[1]]
    shifted1 = np.empty(states)
    shifted2 = np.empty(states)
    for t in range(1, frames):
        shifted1[0] = -np.inf
        shifted1[1:] = alpha[:-1]
        shifted2[:2] = -np.inf
        shifted2[2:] = np.where(can_skip[2:], alpha[:-2], -np.inf)
        alpha = np.logaddexp(np.logaddexp(alpha, shifted1), shifted2) + lp[t, ext]
    total = float(np.logaddexp(alpha[-1], alpha[-2]))
    return total if np.isfinite(total) else None


@dataclass
class TextScore:
    #: log P(text | audio window), best over the text's number readings
    log_likelihood: float
    #: tokens in the best reading (0 for the empty text)
    tokens: int


class PassageScorer:
    """Scores alternative wordings of one passage against its audio."""

    def __init__(self, model: CTCModel, language: str | None):
        self.model = model
        self.language = language
        self._numbers = isinstance(model, NemoCTCModel)  # English model: English number readings

    def supports(self) -> bool:
        return self.model.supports(self.language)

    def emissions(self, audio: np.ndarray) -> np.ndarray:
        """Per-frame log-probabilities for an audio window (level-normalised,
        as for alignment, so quiet speakers are heard)."""
        return self.model.emissions(even_out(audio))

    def word_tokens(self, word: str) -> list[list[int]] | None:
        """Token sequences the word may be spoken as (several for numbers), or
        None if the model cannot spell it."""
        tokens = self.model.tokenize(word, self.language)
        if tokens:
            return [tokens]
        if not self._numbers:
            return None
        token = word.strip().strip(".,;:!?\"()[]")
        options = readings(token)
        if options is None:
            return None
        spelled: list[list[int]] = []
        for option in options:
            pieces: list[int] = []
            for w in option:
                t = self.model.tokenize(w, self.language)
                if not t:
                    break
                pieces.extend(t)
            else:
                spelled.append(pieces)
        return spelled or None

    def spellable(self, words: Sequence[str]) -> bool:
        return all(self.word_tokens(w) is not None for w in words if w.strip())

    def score(self, emissions: np.ndarray, words: Sequence[str]) -> TextScore | None:
        """Best score of the words over their readings, or None if a word
        cannot be spelled or the window is too short for the text."""
        per_word: list[list[list[int]]] = []
        for word in words:
            if not word.strip():
                continue
            options = self.word_tokens(word)
            if options is None:
                return None
            per_word.append(options)
        best: TextScore | None = None
        for combination in itertools.islice(itertools.product(*per_word), MAX_COMBINATIONS):
            targets = [token for option in combination for token in option]
            value = ctc_log_likelihood(emissions, targets, self.model.blank)
            if value is not None and (best is None or value > best.log_likelihood):
                best = TextScore(value, len(targets))
        return best
