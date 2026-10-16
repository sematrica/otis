"""Attach diarization to the transcript: a speaker for every word and segment.

Each word takes the speaker who is active for most of its duration. A
recognised segment that contains a speaker change is split at the change, so
that every segment has exactly one speaker; the words themselves are never
altered, only grouped differently, and each split is recorded in the
segments' flags.

Assignments the evidence does not clearly support are flagged rather than
hidden:

* ``speaker_uncertain`` - the chosen speaker covers little of the speech,
  overlapping analysis windows disagree, or the voice is nearly as close to
  another speaker's;
* ``overlapping_speech`` - two or more people talk at once in the segment;
* ``speaker_change_uncertain`` - a word next to a speaker change may belong
  to the other speaker (the change could be off by a word);
* ``speaker_unassigned`` - diarization found no speech near the words.

The thresholds are heuristics (documented in ``docs/ARCHITECTURE.md``), not
calibrated probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..transcript.model import TSegment, TWord
from .pipeline import DiarizationResult

#: Look this far (seconds) for a speaker when no one is active under a word.
NEAREST_SPEAKER_MAX_GAP = 1.0
#: A word shorter than this is widened to this length before looking up who
#: speaks (alignment and DTW timings can be very short).
MIN_WORD_SPAN = 0.08
LOW_COVERAGE = 0.5
LOW_AGREEMENT = 0.5
LOW_MARGIN = 0.1
OVERLAP_FRACTION = 0.2
OVERLAP_MIN_SECONDS = 0.3
#: A run of words that switches speaker inside a segment and back again is
#: folded into the surrounding speaker when its own support is this weak.
WEAK_RUN_COVERAGE = 0.5
#: A word next to a speaker change whose speaker covers less of it than this
#: may belong to the other side of the change.
BOUNDARY_COVERAGE = 0.75

#: Flags describing chunk-boundary decisions; they stay with the part of a
#: split segment where the boundary was.
_BOUNDARY_FLAGS = {"boundary_duplicate_removed", "trimmed_at_chunk_boundary", "chunk_boundary"}


@dataclass
class WordEvidence:
    speaker: int | None
    coverage: float  # fraction of the word's frames where `speaker` is active
    agreement: float  # mean activation of `speaker` over those frames
    margin: float | None  # mean embedding margin of `speaker`
    overlap: float  # fraction of frames with 2+ speakers
    others: tuple[int, ...]  # other speakers active during the word
    nearest: bool  # speaker taken from nearby speech, not under the word


def word_evidence(diar: DiarizationResult, start: float, end: float) -> WordEvidence:
    if end - start < MIN_WORD_SPAN:
        centre = (start + end) / 2
        start, end = centre - MIN_WORD_SPAN / 2, centre + MIN_WORD_SPAN / 2
    if diar.num_speakers == 0:
        return WordEvidence(None, 0.0, 0.0, None, 0.0, (), False)
    frames = diar.frame_range(start, end)
    binary = diar.binary[frames]
    n = binary.shape[0]
    if n == 0:
        return WordEvidence(None, 0.0, 0.0, None, 0.0, (), False)
    votes = binary.sum(axis=0).astype(np.float64)
    if votes.max() > 0:
        weight = (diar.activation[frames] * binary).sum(axis=0)
        speaker = int(np.lexsort((-weight, -votes))[0])
        mask = binary[:, speaker] > 0
        margins = diar.margin[frames][mask, speaker]
        margins = margins[np.isfinite(margins)]
        return WordEvidence(
            speaker=speaker,
            coverage=float(votes[speaker] / n),
            agreement=float(diar.activation[frames][mask, speaker].mean()),
            margin=float(margins.mean()) if margins.size else None,
            overlap=float((binary.sum(axis=1) >= 2).mean()),
            others=tuple(int(i) for i in np.nonzero(votes)[0] if i != speaker),
            nearest=False,
        )
    speaker = _nearest_speaker(diar, start, end)
    return WordEvidence(speaker, 0.0, 0.0, None, 0.0, (), speaker is not None)


def _nearest_speaker(diar: DiarizationResult, start: float, end: float) -> int | None:
    window = diar.frame_range(start - NEAREST_SPEAKER_MAX_GAP, end + NEAREST_SPEAKER_MAX_GAP)
    active = diar.binary[window].any(axis=1)
    if not active.any():
        return None
    centre = diar.frame_range(start, end)
    mid = (centre.start + centre.stop) / 2 - window.start
    idx = np.nonzero(active)[0]
    nearest = int(window.start + idx[np.argmin(np.abs(idx - mid))])
    candidates = np.nonzero(diar.binary[nearest])[0]
    return int(candidates[np.argmax(diar.activation[nearest, candidates])])


def assign_speakers(segments: Sequence[TSegment], diar: DiarizationResult) -> list[TSegment]:
    """New segments (split at speaker changes) with speakers on segments and words."""
    out: list[TSegment] = []
    multi = diar.num_speakers >= 2
    for seg in segments:
        if not seg.words:
            ev = word_evidence(diar, seg.start, seg.end)
            out.append(_finish(seg, seg.words, [ev], [ev.speaker], multi, seg.start, seg.end, seg.text, [], True))
            continue
        evidence = [word_evidence(diar, w.start, w.end) for w in seg.words]
        labels = _fold_weak_runs(_fill_unknown([e.speaker for e in evidence]), evidence)
        runs = _runs(labels)
        if len(runs) == 1:
            out.append(_finish(seg, seg.words, evidence, labels, multi, seg.start, seg.end, seg.text, [], True))
            continue
        for part, (a, b) in enumerate(runs):
            words = seg.words[a:b]
            flag = {
                "type": "split_at_speaker_change",
                "part": part + 1,
                "parts": len(runs),
                "original_start": round(seg.start, 3),
                "original_end": round(seg.end, 3),
            }
            start = seg.start if part == 0 else words[0].start
            end = seg.end if part == len(runs) - 1 else words[-1].end
            text = "".join(w.text for w in words)
            extra = [flag]
            edges = ([("start", a)] if part > 0 else []) + ([("end", b - 1)] if part < len(runs) - 1 else [])
            shaky = [(where, i) for where, i in edges
                     if evidence[i].coverage < BOUNDARY_COVERAGE or evidence[i].overlap > 0 or evidence[i].speaker != labels[i]]
            if shaky:
                extra.append({
                    "type": "speaker_change_uncertain",
                    "words": [{"position": where, "text": seg.words[i].text.strip(),
                               "coverage": round(evidence[i].coverage, 3)} for where, i in shaky],
                })
            out.append(_finish(seg, words, evidence[a:b], labels[a:b], multi, start, end, text, extra, part == 0))
    return out


def _fill_unknown(labels: list[int | None]) -> list[int | None]:
    """Words with no speaker evidence join the preceding (else following) word."""
    out = list(labels)
    last = None
    for i, value in enumerate(out):
        if value is None:
            out[i] = last
        else:
            last = value
    following = None
    for i in range(len(out) - 1, -1, -1):
        if out[i] is None:
            out[i] = following
        else:
            following = out[i]
    return out


def _fold_weak_runs(labels: list[int | None], evidence: list[WordEvidence]) -> list[int | None]:
    out = list(labels)
    runs = _runs(out)
    for i in range(1, len(runs) - 1):
        a, b = runs[i]
        before, after = out[runs[i - 1][0]], out[runs[i + 1][0]]
        if before is None or before != after:
            continue
        if np.mean([evidence[j].coverage for j in range(a, b)]) < WEAK_RUN_COVERAGE:
            for j in range(a, b):
                out[j] = before
    return out


def _runs(labels: list[int | None]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((start, i))
            start = i
    return runs


def _finish(
    seg: TSegment,
    words: list[TWord],
    evidence: list[WordEvidence],
    labels: list[int | None],
    multi: bool,
    start: float,
    end: float,
    text: str,
    extra_flags: list[dict],
    keep_boundary_flags: bool,
) -> TSegment:
    speaker = labels[0] if labels else None
    flags = [dict(f) for f in seg.flags if keep_boundary_flags or f.get("type") not in _BOUNDARY_FLAGS]
    flags.extend(extra_flags)
    new_words = [TWord(**{**w.__dict__, "speaker": label}) for w, label in zip(words, labels)]
    result = TSegment(
        start=start, end=end, text=text, words=new_words, language=seg.language,
        avg_logprob=seg.avg_logprob, no_speech_prob=seg.no_speech_prob, compression_ratio=seg.compression_ratio,
        temperature=seg.temperature, chunk_index=seg.chunk_index, flags=flags, speaker=speaker,
    )
    if speaker is None:
        flags.append({"type": "speaker_unassigned", "reason": "no speech found by diarization near these words"})
        return result

    durations = np.array([max(1e-3, w.end - w.start) for w in words] if words else [max(1e-3, end - start)])
    weights = durations / durations.sum()
    coverage = float(sum(w * (e.coverage if e.speaker == speaker else 0.0) for w, e in zip(weights, evidence)))
    agreement = float(sum(w * (e.agreement if e.speaker == speaker else 0.0) for w, e in zip(weights, evidence)))
    overlap = np.array([e.overlap for e in evidence])
    margins = [e.margin for e in evidence if e.speaker == speaker and e.margin is not None]
    margin = float(np.mean(margins)) if margins else None
    result.speaker_confidence = round(coverage * agreement, 3)

    reasons = []
    if any(e.nearest for e in evidence):
        reasons.append("no_speech_under_words")
    if coverage < LOW_COVERAGE:
        reasons.append("low_coverage")
    if agreement < LOW_AGREEMENT:
        reasons.append("windows_disagree")
    if multi and margin is not None and margin < LOW_MARGIN:
        reasons.append("similar_to_other_speaker")
    if reasons:
        flags.append(
            {
                "type": "speaker_uncertain",
                "reasons": reasons,
                "coverage": round(coverage, 3),
                "agreement": round(agreement, 3),
                "margin": round(margin, 3) if margin is not None else None,
            }
        )
    overlap_fraction = float((overlap * weights).sum())
    overlap_seconds = float((overlap * durations).sum())
    if overlap_fraction >= OVERLAP_FRACTION or overlap_seconds >= OVERLAP_MIN_SECONDS:
        flags.append(
            {
                "type": "overlapping_speech",
                "fraction": round(overlap_fraction, 3),
                "seconds": round(overlap_seconds, 3),
                "other_speakers": sorted({o for e in evidence for o in e.others if o != speaker}),
            }
        )
    return result
