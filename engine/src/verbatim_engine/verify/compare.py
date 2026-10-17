"""Pass 7: choose between transcription candidates by acoustic evidence.

Every candidate wording of a passage (the first pass and the re-transcriptions)
is scored with the CTC acoustic model on the same audio window, between the
same *anchor* words (the transcript's words just before and after the
passage, which all candidates share). The anchors pin the window's edges, so
a candidate cannot win by quietly dropping a word at the boundary.

Decisions are deliberately conservative:

* the first-pass wording is kept unless another candidate beats it by
  :data:`DECISION_MARGIN` (log-likelihood, nats), plus :data:`DELETION_COST`
  for every word the candidate has fewer than the first pass;
* wordings that differ but are within the margin are kept as *unresolved
  alternatives* - agreement or disagreement between decodes is not taken as
  evidence either way;
* if "nobody said anything here" explains the audio better than every
  candidate by the margin, the passage is reported as lacking acoustic
  support (a possible hallucination). Its words are **not** removed;
* for a stretch of untranscribed speech (empty first pass), words are
  recovered only if they beat the empty hypothesis by the margin.

The margin was set from measurements on real speech (see the Stage 3
report): single-word substitutions, insertions and repetitions of the true
text always scored lower than the truth (by at least 3.5 nats), while
dropping a short, quickly spoken word occasionally scored up to 1.2 nats
*better* than the truth. A margin of 3 nats keeps such deletions from ever
replacing a correct wording.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..alignment.score import PassageScorer
from ..asr.merge import normalize_token

DECISION_MARGIN = 3.0
#: Extra lead required per word a candidate has fewer than the first pass.
#: CTC scoring favours dropping words where the acoustic model is unsure
#: (far-field, overlapping speech): on the AMI meeting excerpts, wordings that
#: won by dropping words left reference speech uncovered (worse DER). Each
#: dropped word must be paid for with a clear margin of its own.
DELETION_COST = 3.0


@dataclass
class Candidate:
    #: "first_pass" or the name of the decoding configuration
    source: str
    words: list[str]

    @property
    def key(self) -> tuple[str, ...]:
        return tuple(t for t in (normalize_token(w) for w in self.words) if t)


@dataclass
class CandidateScore:
    source: str
    text: str
    log_likelihood: float | None  # None: the acoustic model cannot spell it


@dataclass
class Comparison:
    #: "keep" | "replace" | "unresolved" | "no_acoustic_support" | "unscorable"
    decision: str
    #: index into the candidates of the wording to use
    chosen: int
    #: lead of the chosen wording over the first pass (replace), or of the
    #: first pass over the best alternative
    margin: float | None
    scores: list[CandidateScore] = field(default_factory=list)
    #: log-likelihood of "no words here"
    empty: float | None = None
    #: indices of distinct wordings within the margin of the chosen one
    alternatives: list[int] = field(default_factory=list)


def compare(
    scorer: PassageScorer,
    emissions: np.ndarray,
    candidates: Sequence[Candidate],
    *,
    left: Sequence[str] = (),
    right: Sequence[str] = (),
) -> Comparison:
    """Compare candidate wordings; ``candidates[0]`` is the first pass."""
    if not candidates:
        raise ValueError("no candidates")
    distinct: dict[tuple[str, ...], float | None] = {}
    for cand in candidates:
        if cand.key not in distinct:
            result = scorer.score(emissions, [*left, *cand.words, *right])
            distinct[cand.key] = None if result is None else result.log_likelihood
    empty_result = scorer.score(emissions, [*left, *right])
    empty = None if empty_result is None else empty_result.log_likelihood
    scores = [CandidateScore(c.source, " ".join(c.words), distinct[c.key]) for c in candidates]

    first = distinct[candidates[0].key]
    # An empty alternative never replaces words: "nothing was said" is judged
    # separately (no_acoustic_support), and words are never removed.
    others = [i for i, c in enumerate(candidates)
              if c.key != candidates[0].key and distinct[c.key] is not None and (c.key or not candidates[0].key)]
    # one representative per distinct wording, best first
    seen: set[tuple[str, ...]] = set()
    ranked: list[int] = []
    for i in sorted(others, key=lambda i: -distinct[candidates[i].key]):  # type: ignore[operator]
        if candidates[i].key not in seen:
            seen.add(candidates[i].key)
            ranked.append(i)

    if first is None:
        return Comparison("unscorable", 0, None, scores, empty, ranked)
    best_text = max([first] + [distinct[candidates[i].key] for i in ranked])  # type: ignore[list-item]
    if candidates[0].key and empty is not None and empty - best_text >= DECISION_MARGIN:
        return Comparison("no_acoustic_support", 0, round(empty - best_text, 3), scores, empty, ranked)
    if ranked:
        best = ranked[0]
        lead = distinct[candidates[best].key] - first  # type: ignore[operator]
        dropped = max(0, len(candidates[0].key) - len(candidates[best].key))
        if lead >= DECISION_MARGIN + DELETION_COST * dropped:
            close = [i for i in ranked[1:] if distinct[candidates[best].key] - distinct[candidates[i].key] < DECISION_MARGIN]
            return Comparison("replace", best, round(lead, 3), scores, empty, close)
        close = [i for i in ranked if first - distinct[candidates[i].key] < DECISION_MARGIN]  # type: ignore[operator]
        if close:
            return Comparison("unresolved", 0, round(first - distinct[candidates[ranked[0]].key], 3), scores, empty, close)
        return Comparison("keep", 0, round(first - distinct[candidates[ranked[0]].key], 3), scores, empty)
    return Comparison("keep", 0, None, scores, empty)
