"""Pass 8: validate the final transcript.

Checks, after every other pass has changed what it changes:

* **timestamps** - inside the recording, a word's end not before its start,
  segment bounds covering their words, words in order. Out-of-range or
  inverted times are corrected (and counted); words out of order are only
  flagged, since reordering could change what was said;
* **duplication** - the same run of words repeated across a segment boundary
  within a moment (``possible_duplicate``; nothing is removed);
* **missing segments** - detected speech that no word covers, recomputed on
  the final transcript (``missing_speech``), so recovered passages no longer
  count and unresolved ones stay flagged.

The result summarises what still needs review.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..asr.merge import normalize_token
from ..transcript.model import TSegment
from .quality import SpeechMap, flag_missing_speech
from .review import LEVELS, review_priority

#: A repeated word run at least this long, this close in time, is a likely duplicate.
DUPLICATE_MIN_WORDS = 3
DUPLICATE_MAX_GAP = 1.0
#: Words may overlap their predecessor this much before being "out of order".
ORDER_TOLERANCE = 0.05


def validate(segments: list[TSegment], speech: SpeechMap | None, duration: float) -> tuple[list[TSegment], dict[str, Any]]:
    out = sorted(segments, key=lambda s: (s.start, s.end))
    fixes = 0
    disorder = 0
    for seg in out:
        for word in seg.words:
            start, end = min(max(0.0, word.start), duration), min(max(0.0, word.end), duration)
            if end < start:
                end = start
            if (start, end) != (word.start, word.end):
                word.start, word.end = start, end
                fixes += 1
        if seg.words:
            out_of_order = sum(1 for a, b in zip(seg.words, seg.words[1:]) if b.start < a.start - ORDER_TOLERANCE)
            if out_of_order:
                seg.flags.append({"type": "timing_inconsistent", "severity": "medium", "words": out_of_order})
                disorder += 1
            start = min(seg.start, seg.words[0].start)
            end = max(seg.end, max(w.end for w in seg.words))
        else:
            start, end = seg.start, seg.end
        start, end = min(max(0.0, start), duration), min(max(0.0, end), duration)
        end = max(end, start)
        if (start, end) != (seg.start, seg.end):
            seg.start, seg.end = start, end
            fixes += 1

    duplicates = 0
    for prev, seg in zip(out, out[1:]):
        found = _boundary_repeat(prev, seg)
        if found is not None:
            seg.flags.append({"type": "possible_duplicate", "severity": "medium", "text": found})
            duplicates += 1

    for seg in out:
        seg.flags = [f for f in seg.flags if f.get("type") != "missing_speech"]
    gaps = flag_missing_speech(out, speech) if speech is not None else []

    priorities = {"high": 0, "medium": 0, "low": 0}
    for seg in out:
        level = review_priority(seg.flags)
        if level is not None:
            priorities[level] += 1
    summary = {
        "segments": len(out),
        "timing_fixes": fixes,
        "words_out_of_order": disorder,
        "possible_duplicates": duplicates,
        "missing_speech": len(gaps),
        "untranscribed_seconds": round(sum(g["end"] - g["start"] for g in gaps), 2),
        "untranscribed": [{k: g[k] for k in ("start", "end", "severity")} for g in gaps[:MAX_LISTED]],
        "review": priorities,
        "needs_review": priorities["high"] + priorities["medium"],
    }
    return out, summary


#: Stretches of untranscribed speech listed in the summary (all are counted).
MAX_LISTED = 100


def _boundary_repeat(prev: TSegment, seg: TSegment) -> str | None:
    if not prev.words or not seg.words or seg.words[0].start - prev.words[-1].end > DUPLICATE_MAX_GAP:
        return None
    a = [normalize_token(w.text) for w in prev.words]
    b = [normalize_token(w.text) for w in seg.words]
    for k in range(min(len(a), len(b)), DUPLICATE_MIN_WORDS - 1, -1):
        if a[-k:] == b[:k] and all(a[-k:]):
            return " ".join(w.text.strip() for w in seg.words[:k])
    return None


def needs_review_count(segments: Sequence[TSegment]) -> int:
    return sum(1 for seg in segments if LEVELS.get(review_priority(seg.flags) or "", 0) >= LEVELS["medium"])
