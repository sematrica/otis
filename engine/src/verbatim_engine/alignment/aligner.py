"""Align transcript words to the audio, segment group by segment group.

Consecutive segments (up to ``GROUP_MAX_SECONDS``, without long gaps) are
aligned together against their audio plus a margin, so that a slightly
misplaced segment boundary from the recogniser does not matter. A wildcard
token ("star", free to absorb any frames) sits at the edges of each group,
between segments, and in place of words the model cannot spell: it soaks up
speech that belongs to neighbouring groups, was not transcribed, or is a
number, instead of forcing real words onto it.

Every segment's result is checked; implausible results are rejected and the
recogniser's timings kept (``status == "failed"`` with a reason).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from ..audio.level import even_out
from ..audio.wav import WavReader
from ..errors import Cancelled
from . import refine as boundaries
from .ctc import forced_align
from .models import CTCModel

log = logging.getLogger(__name__)

GROUP_MAX_SECONDS = 30.0
GROUP_MAX_GAP_SECONDS = 2.0
GROUP_PAD_SECONDS = 0.6
#: Log-probability of the wildcard token in every frame (0 = always free).
STAR_LOG_PROB = 0.0
#: A segment whose tokens average a lower log-probability than this did not
#: match the audio (e.g. the text is not what was said there).
MIN_MEAN_TOKEN_LOG_PROB = -4.0
#: ... or whose aligned words start, on median, this far from the
#: recogniser's own estimate.
MAX_MEDIAN_SHIFT_SECONDS = 1.5
MAX_WORD_SECONDS = 4.0


class _Word(Protocol):
    start: float
    end: float
    text: str


class _Segment(Protocol):
    start: float
    end: float
    words: Sequence[_Word] | None


@dataclass
class WordTiming:
    start: float
    end: float
    #: mean log-probability of the word's tokens (None for wildcard words)
    score: float | None
    #: "alignment", or "wildcard" for words timed by the wildcard token
    source: str


@dataclass
class SegmentAlignment:
    status: str  # "aligned" | "failed" | "skipped"
    reason: str | None = None
    words: list[WordTiming] | None = None
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "score": self.score,
            "words": [[w.start, w.end, w.score, w.source] for w in self.words] if self.words else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SegmentAlignment":
        words = [WordTiming(*w) for w in data["words"]] if data.get("words") else None
        return cls(data["status"], data.get("reason"), words, data.get("score"))


def plan_groups(segments: Sequence[_Segment]) -> list[list[int]]:
    groups: list[list[int]] = []
    for i, seg in enumerate(segments):
        if not seg.words:
            continue
        if groups:
            first = segments[groups[-1][0]]
            last = segments[groups[-1][-1]]
            if seg.start - last.end <= GROUP_MAX_GAP_SECONDS and _span_end(seg) - _span_start(first) <= GROUP_MAX_SECONDS:
                groups[-1].append(i)
                continue
        groups.append([i])
    return groups


def _span_start(seg: _Segment) -> float:
    return min([seg.start] + [w.start for w in seg.words or []])


def _span_end(seg: _Segment) -> float:
    return max([seg.end] + [w.end for w in seg.words or []])


class WordAligner:
    def __init__(self, model: CTCModel):
        self.model = model

    def align(
        self,
        reader: WavReader,
        segments: Sequence[_Segment],
        language: str | None,
        *,
        done: dict[int, SegmentAlignment] | None = None,
        on_group: Callable[[list[int], dict[int, SegmentAlignment]], None] | None = None,
        progress: Callable[[float, float], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[SegmentAlignment]:
        """Align all segments. ``done`` holds results from an earlier,
        interrupted run (skipped here); ``on_group`` receives each new group's
        results so the caller can persist them."""
        results: dict[int, SegmentAlignment] = dict(done or {})
        groups = plan_groups(segments)
        total = sum(_span_end(segments[g[-1]]) - _span_start(segments[g[0]]) for g in groups) or 1.0
        processed = 0.0
        for group in groups:
            span = _span_end(segments[group[-1]]) - _span_start(segments[group[0]])
            if all(i in results for i in group):
                processed += span
                continue
            if should_cancel is not None and should_cancel():
                raise Cancelled("The job was cancelled.")
            new = self._align_group(reader, segments, group, language)
            results.update(new)
            if on_group is not None:
                on_group(group, new)
            processed += span
            if progress is not None:
                progress(processed, total)
        return [results.get(i, SegmentAlignment("skipped", "no words to align")) for i in range(len(segments))]

    # ------------------------------------------------------------------ one group

    def _align_group(self, reader: WavReader, segments: Sequence[_Segment], group: list[int], language: str | None) -> dict[int, SegmentAlignment]:
        start = max(0.0, _span_start(segments[group[0]]) - GROUP_PAD_SECONDS)
        end = min(reader.duration, _span_end(segments[group[-1]]) + GROUP_PAD_SECONDS)
        if end - start < 0.1:
            return {i: SegmentAlignment("failed", "segment has no audio") for i in group}
        audio = reader.read(start, end)
        try:
            emissions = self.model.emissions(even_out(audio))
        except Exception as exc:  # noqa: BLE001 - model runtime errors
            log.warning("alignment model failed on %.1f-%.1f s: %s", start, end, exc)
            return {i: SegmentAlignment("failed", f"alignment model error: {exc}") for i in group}
        star = emissions.shape[1]
        lp = np.concatenate([emissions, np.full((emissions.shape[0], 1), STAR_LOG_PROB, dtype=np.float32)], axis=1)

        targets: list[int] = []
        owners: list[tuple[int, int] | None] = []
        wildcard_words: set[tuple[int, int]] = set()

        def add_star(owner: tuple[int, int] | None) -> None:
            if owner is None and targets and targets[-1] == star:
                return
            targets.append(star)
            owners.append(owner)

        add_star(None)
        for i in group:
            for j, word in enumerate(segments[i].words or []):
                tokens = self.model.tokenize(word.text, language)
                if tokens:
                    targets.extend(tokens)
                    owners.extend([(i, j)] * len(tokens))
                else:
                    wildcard_words.add((i, j))
                    add_star((i, j))
            add_star(None)

        if all(o is None or o in wildcard_words for o in owners):
            return {i: SegmentAlignment("skipped", "no words the alignment model can spell") for i in group}
        alignment = forced_align(lp, targets, self.model.blank)
        if alignment is None:
            if len(group) > 1:  # the text may not fit the joint window: try one segment at a time
                out: dict[int, SegmentAlignment] = {}
                for i in group:
                    out.update(self._align_group(reader, segments, [i], language))
                return out
            return {i: SegmentAlignment("failed", "the audio is too short for the text") for i in group}

        fs = self.model.frame_seconds
        first: dict[tuple[int, int], int] = {}
        last: dict[tuple[int, int], int] = {}
        scores: dict[tuple[int, int], list[float]] = {}
        for span in alignment.spans:
            owner = owners[span.target_index]
            if owner is None:
                continue
            first.setdefault(owner, span.first_frame)
            last[owner] = span.last_frame
            if owner not in wildcard_words:
                scores.setdefault(owner, []).append(span.mean_log_prob)

        keys = [(i, j) for i in group for j in range(len(segments[i].words or []))]
        spans = [boundaries.Span(start + first[k] * fs, start + (last[k] + 1) * fs) for k in keys]
        split = getattr(self.model, "connected_split", boundaries.CONNECTED_SPLIT)
        refined = dict(zip(keys, boundaries.refine(spans, audio, start, split)))
        out = {}
        for i in group:
            words = segments[i].words or []
            timings: list[WordTiming] = []
            for j in range(len(words)):
                key = (i, j)
                t0, t1 = refined[key].start, refined[key].end
                if key in wildcard_words:
                    timings.append(WordTiming(round(t0, 3), round(t1, 3), None, "wildcard"))
                else:
                    timings.append(WordTiming(round(t0, 3), round(t1, 3), round(float(np.mean(scores[key])), 3), "alignment"))
            out[i] = self._check(words, timings)
        return out

    @staticmethod
    def _check(words: Sequence[_Word], timings: list[WordTiming]) -> SegmentAlignment:
        scored = [t.score for t in timings if t.score is not None]
        if not scored:
            return SegmentAlignment("skipped", "no words the alignment model can spell")
        score = round(float(np.mean(scored)), 3)
        if score < MIN_MEAN_TOKEN_LOG_PROB:
            return SegmentAlignment("failed", "the audio does not match the text well enough", score=score)
        shifts = [abs(t.start - w.start) for t, w in zip(timings, words) if t.source == "alignment"]
        if shifts and float(np.median(shifts)) > MAX_MEDIAN_SHIFT_SECONDS:
            return SegmentAlignment("failed", "alignment disagrees with the recogniser's timing", score=score)
        if any(t.end - t.start > MAX_WORD_SECONDS for t in timings if t.source == "alignment"):
            return SegmentAlignment("failed", "implausibly long word", score=score)
        return SegmentAlignment("aligned", None, timings, score)
