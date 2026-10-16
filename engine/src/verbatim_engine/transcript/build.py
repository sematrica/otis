"""Combine recognition, alignment and diarization results into the stored
transcript."""

from __future__ import annotations

from typing import Any, Sequence

from ..alignment.aligner import SegmentAlignment
from .model import TSegment


def apply_alignment(segments: list[TSegment], results: Sequence[SegmentAlignment | None]) -> dict[str, Any]:
    """Move word times to their aligned positions, in place.

    Segments whose alignment failed keep the recogniser's word and segment
    times and are flagged ``alignment_failed`` with the reason. Whisper's own
    times stay on every word (``whisper_start``/``whisper_end``).
    """
    summary = {"segments": 0, "aligned": 0, "failed": 0, "skipped": 0, "wildcard_words": 0}
    for seg, result in zip(segments, results):
        if result is None or not seg.words:
            continue
        summary["segments"] += 1
        if result.status == "aligned" and result.words and len(result.words) == len(seg.words):
            for word, timing in zip(seg.words, result.words):
                word.start, word.end = timing.start, timing.end
                word.timing_source = timing.source
                word.alignment_score = timing.score
                if timing.source == "wildcard":
                    summary["wildcard_words"] += 1
            # the segment spans its words (the recogniser's segment edges
            # usually include surrounding silence)
            seg.start = min(seg.words[0].start, seg.end)
            seg.end = max(seg.words[-1].end, seg.start)
            summary["aligned"] += 1
        elif result.status == "failed":
            seg.flags.append({"type": "alignment_failed", "reason": result.reason, "score": result.score})
            summary["failed"] += 1
        else:
            summary["skipped"] += 1
    return summary


def speakers_in_order(segments: Sequence[TSegment]) -> list[dict[str, Any]]:
    """Speakers that appear in the transcript, labelled by first appearance.

    Returns ``[{"index": diarization index, "label": "Speaker N"}]``. Speakers
    the diarization found but who have no transcribed words (laughter, music,
    untranscribed speech) are left out.
    """
    order: list[int] = []
    for seg in sorted(segments, key=lambda s: s.start):
        if seg.speaker is not None and seg.speaker not in order:
            order.append(seg.speaker)
    return [{"index": index, "label": f"Speaker {n}", "color_index": n - 1} for n, index in enumerate(order, start=1)]
