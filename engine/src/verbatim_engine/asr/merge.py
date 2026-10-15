"""Merge per-chunk recognition results into one transcript.

When a cut lies in a pause, the chunk planner gives each chunk audio that
stops at the edges of that pause, so neighbouring chunks never hear the same
speech: their words are concatenated unchanged (boundary method ``pause``).

When a cut had to be placed inside continuous speech, neighbouring chunks
overlap by ``pad`` seconds on each side, so the *overlap zone*
``[cut - pad, cut + pad]`` is decoded twice. For such a cut:

1. **Alignment (preferred).** The words both chunks produced in the zone are
   aligned as token sequences. If they agree on a word near the cut (same
   normalised text, consistent timing), the transcript is spliced there:
   earlier words come from the left chunk, that word and later ones from the
   right chunk. Nothing is lost and nothing is duplicated.
2. **Fallback (chunks disagree).** The left chunk keeps zone words whose
   midpoint is before the cut; the right chunk keeps words that *end* after
   the left chunk's last kept word *starts*. (Start times are not used:
   Whisper stretches the first word of a window back to the window start.)
   This rule does not drop speech, but can duplicate; so a conservative duplicate check
   then removes an identical word sequence only if both copies lie in the zone
   less than ``MAX_DUPLICATE_GAP_S`` apart — something a genuine repetition at
   a cut placed in a pause cannot satisfy.

Every boundary decision is recorded in the affected segments' ``flags``
(including the full text of anything removed), so it stays reviewable.
Segments without word timings fall back to "kept by the chunk it starts in".
"""

from __future__ import annotations

import difflib
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from ..audio.chunking import Chunk
from .base import Segment, TranscriptionResult, Word

MIN_DUPLICATE_WORDS = 2
MAX_DUPLICATE_WORDS = 16
MAX_DUPLICATE_GAP_S = 0.4
#: Allowed timing disagreement for an aligned word, by matching-block length.
ALIGN_TOLERANCE_SHORT_S = 0.6
ALIGN_TOLERANCE_LONG_S = 1.5
ALIGN_LONG_BLOCK = 3


@dataclass
class MergedSegment:
    start: float
    end: float
    text: str
    words: list[Word] | None
    avg_logprob: float | None
    no_speech_prob: float | None
    compression_ratio: float | None
    temperature: float | None
    chunk_index: int
    language: str | None
    flags: list[dict[str, Any]] = field(default_factory=list)


def normalize_token(token: str) -> str:
    stripped = "".join(ch for ch in token if not unicodedata.category(ch).startswith("P"))
    return stripped.strip().casefold()


def _mid(w: Word) -> float:
    return (w.start + w.end) / 2.0


@dataclass
class _ChunkWords:
    chunk: Chunk
    result: TranscriptionResult
    segments: list[Segment]                 # shifted to absolute time
    flat: list[tuple[int, int, Word]]       # (segment index, word index, word)
    keep_from: int = 0                      # flat index range kept
    keep_to: int = -1
    timed: bool = True                      # all segments have word timings
    flags: dict[int, list[dict[str, Any]]] = field(default_factory=dict)  # per segment

    def __post_init__(self) -> None:
        if self.keep_to < 0:
            self.keep_to = len(self.flat)

    def flag(self, segment_index: int, flag: dict[str, Any]) -> None:
        self.flags.setdefault(segment_index, []).append(flag)


def _prepare(chunk: Chunk, result: TranscriptionResult) -> _ChunkWords:
    segments = [s.shifted(chunk.audio_start) for s in result.segments]
    flat = [(si, wi, w) for si, s in enumerate(segments) for wi, w in enumerate(s.words or [])]
    timed = all(s.words for s in segments if s.text.strip())
    return _ChunkWords(chunk, result, segments, flat, timed=timed)


def _splice(left: _ChunkWords, right: _ChunkWords, cut: float, pad: float) -> None:
    zone_left = [i for i in range(left.keep_from, left.keep_to) if _mid(left.flat[i][2]) >= cut - pad]
    zone_right = [i for i in range(right.keep_from, right.keep_to) if _mid(right.flat[i][2]) <= cut + pad]
    if not zone_left or not zone_right:
        # Only one chunk heard anything in the overlap: its words are the only
        # evidence, so they are all kept.
        return

    a = [normalize_token(left.flat[i][2].text) for i in zone_left]
    b = [normalize_token(right.flat[i][2].text) for i in zone_right]
    best: tuple[int, float, int, int] | None = None  # (block size, -distance to cut, ai, bj)
    for block in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        if block.size == 0:
            continue
        tolerance = ALIGN_TOLERANCE_LONG_S if block.size >= ALIGN_LONG_BLOCK else ALIGN_TOLERANCE_SHORT_S
        for k in range(block.size):
            ai, bj = block.a + k, block.b + k
            if not a[ai]:
                continue
            wl, wr = left.flat[zone_left[ai]][2], right.flat[zone_right[bj]][2]
            if abs(_mid(wl) - _mid(wr)) > tolerance:
                continue
            distance = abs((_mid(wl) + _mid(wr)) / 2.0 - cut)
            candidate = (min(block.size, ALIGN_LONG_BLOCK), -distance, ai, bj)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is not None:
        _, _, ai, bj = best
        left.keep_to = zone_left[ai]
        right.keep_from = zone_right[bj]
        method = "aligned"
    else:
        cut_index = next((i for i in zone_left if _mid(left.flat[i][2]) >= cut), left.keep_to)
        left.keep_to = cut_index
        last_start = left.flat[cut_index - 1][2].start if cut_index > left.keep_from else cut - pad
        right.keep_from = next((i for i in zone_right if right.flat[i][2].end > last_start + 0.05), right.keep_to)
        method = "time_fallback"
        _remove_duplicates(left, right, cut, pad)
    if right.keep_from < right.keep_to:
        right.flag(right.flat[right.keep_from][0], {"type": "chunk_boundary", "cut": round(cut, 3), "method": method})


def _remove_duplicates(left: _ChunkWords, right: _ChunkWords, cut: float, pad: float) -> None:
    tail = [i for i in range(left.keep_from, left.keep_to) if left.flat[i][2].start >= cut - pad]
    head = [i for i in range(right.keep_from, right.keep_to) if right.flat[i][2].end <= cut + pad]
    for k in range(min(len(tail), len(head), MAX_DUPLICATE_WORDS), MIN_DUPLICATE_WORDS - 1, -1):
        ta = [normalize_token(left.flat[i][2].text) for i in tail[-k:]]
        hb = [normalize_token(right.flat[i][2].text) for i in head[:k]]
        if ta != hb or not all(ta):
            continue
        gap = right.flat[head[0]][2].start - left.flat[tail[-1]][2].end
        if gap >= MAX_DUPLICATE_GAP_S:
            return
        removed = [right.flat[i][2] for i in head[:k]]
        right.flag(
            right.flat[head[0]][0],
            {
                "type": "boundary_duplicate_removed",
                "text": "".join(w.text for w in removed).strip(),
                "start": removed[0].start,
                "end": removed[-1].end,
                "duplicate_of_chunk": left.chunk.index,
            },
        )
        right.keep_from = head[k - 1] + 1
        return


def _build(cw: _ChunkWords, is_first: bool, is_last: bool) -> list[MergedSegment]:
    out: list[MergedSegment] = []
    kept_by_segment: dict[int, list[Word]] = {}
    for i in range(cw.keep_from, cw.keep_to):
        si, _wi, w = cw.flat[i]
        kept_by_segment.setdefault(si, []).append(w)
    language = cw.result.language
    pending_flags: list[dict[str, Any]] = []
    for si, seg in enumerate(cw.segments):
        flags = pending_flags + list(cw.flags.get(si, []))
        pending_flags = []
        if seg.words:
            kept = kept_by_segment.get(si, [])
            if not kept:
                pending_flags = flags  # carried to the next surviving segment
                continue
            trimmed = len(kept) != len(seg.words)
            if trimmed:
                text, start, end = "".join(w.text for w in kept).strip(), kept[0].start, kept[-1].end
                flags.append({"type": "trimmed_at_chunk_boundary", "chunk": cw.chunk.index})
            else:
                text, start, end = seg.text.strip(), seg.start, seg.end
            words: list[Word] | None = kept
        else:
            # No word timings: keep the segment in the chunk where it starts.
            starts_here = (is_first or seg.start >= cw.chunk.owned_start) and (is_last or seg.start < cw.chunk.owned_end)
            if not starts_here:
                continue
            text, start, end, words = seg.text.strip(), seg.start, seg.end, None if seg.words is None else []
        if not text:
            pending_flags = flags
            continue
        out.append(
            MergedSegment(
                start=start,
                end=max(end, start),
                text=text,
                words=words,
                avg_logprob=seg.avg_logprob,
                no_speech_prob=seg.no_speech_prob,
                compression_ratio=seg.compression_ratio,
                temperature=seg.temperature,
                chunk_index=cw.chunk.index,
                language=language,
                flags=flags,
            )
        )
    if pending_flags and out:
        out[-1].flags.extend(pending_flags)
    return out


def apply_ownership(chunk: Chunk, result: TranscriptionResult, is_last: bool) -> list[MergedSegment]:
    """Words of one chunk restricted to its owned interval (used to build the
    text prompt for the following chunk)."""
    cw = _prepare(chunk, result)
    kept = [
        i for i, (_si, _wi, w) in enumerate(cw.flat)
        if (chunk.index == 0 or _mid(w) >= chunk.owned_start) and (is_last or _mid(w) < chunk.owned_end)
    ]
    if kept:
        cw.keep_from, cw.keep_to = kept[0], kept[-1] + 1
    else:
        cw.keep_from = cw.keep_to = 0
    return _build(cw, is_first=chunk.index == 0, is_last=is_last)


def assemble(results: list[tuple[Chunk, TranscriptionResult]], pad: float) -> list[MergedSegment]:
    ordered = sorted(results, key=lambda pair: pair[0].index)
    prepared = [_prepare(c, r) for c, r in ordered]
    for left, right in zip(prepared, prepared[1:]):
        cut = right.chunk.owned_start
        if right.chunk.cut_energy_db is not None and not right.chunk.cut_in_speech:
            # Cut in a pause: the chunks' audio did not share any speech.
            if right.flat:
                right.flag(right.flat[0][0], {"type": "chunk_boundary", "cut": round(cut, 3), "method": "pause"})
            elif right.segments:
                right.flag(0, {"type": "chunk_boundary", "cut": round(cut, 3), "method": "pause"})
            continue
        if left.timed and right.timed:
            _splice(left, right, cut, pad)
        else:
            # Mixed timing information: fall back to midpoint ownership for the
            # side that has word timings (untimed segments use their start).
            if left.timed:
                left.keep_to = next((i for i, (_s, _w, w) in enumerate(left.flat) if _mid(w) >= cut), left.keep_to)
            if right.timed:
                right.keep_from = next((i for i, (_s, _w, w) in enumerate(right.flat) if _mid(w) >= cut), right.keep_to)
    merged: list[MergedSegment] = []
    for position, cw in enumerate(prepared):
        merged.extend(_build(cw, is_first=position == 0, is_last=position == len(prepared) - 1))
    merged.sort(key=lambda s: (s.start, s.chunk_index))
    return merged
