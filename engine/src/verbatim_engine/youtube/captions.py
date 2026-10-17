"""Captions published with a video: parsing, and comparison with the transcript.

Captions are a separate, labelled source. They are never merged into the
transcript. Where creator captions or the platform's automatic captions
disagree clearly with a transcribed passage, the passage gets a low-severity
``captions_differ`` flag showing the caption wording, so the user can listen
and decide. Agreement is not taken as proof either way: automatic captions
come from another speech recogniser, and creators' captions are often
condensed.
"""

from __future__ import annotations

import difflib
import html
import re
from dataclasses import dataclass
from typing import Any, Sequence

from ..asr.merge import normalize_token

_TIME = r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})"
_TIMING = re.compile(rf"^\s*{_TIME}\s+-->\s+{_TIME}")
_INLINE_TIME = re.compile(r"<\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3}>")
_TAG = re.compile(r"<[^>]*>")
#: passages compared need at least this many words
MIN_WORDS = 3
#: word-sequence agreement below which a passage is flagged
AGREEMENT = 0.6
#: caption words this far outside a passage still count as its own
SLACK_SECONDS = 0.3


@dataclass
class Cue:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "text": self.text}


def _seconds(h, m, s, ms) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _clean(line: str) -> str:
    return " ".join(html.unescape(_TAG.sub("", line)).replace("‎", "").replace("‏", "").split())


def parse_vtt(text: str) -> list[Cue]:
    """Cues of a WebVTT file. In "rolling" automatic captions (lines whose
    new words carry inline timestamps, each cue repeating the previous line),
    only the newly spoken words of each cue are kept."""
    # Cue blocks end at an empty line. A line holding only spaces belongs to
    # the cue: YouTube's automatic captions start every cue with one.
    blocks = re.split(r"\n{2,}", text.replace("\r\n", "\n").replace("\r", "\n"))
    parsed: list[tuple[float, float, list[str]]] = []
    for block in blocks:
        lines = block.split("\n")
        for i, line in enumerate(lines):
            match = _TIMING.match(line)
            if match:
                g = match.groups()
                parsed.append((_seconds(*g[:4]), _seconds(*g[4:]), lines[i + 1:]))
                break
    rolling = any(_INLINE_TIME.search(line) for _, _, payload in parsed for line in payload)
    cues: list[Cue] = []
    for start, end, payload in parsed:
        lines = [line for line in payload if _INLINE_TIME.search(line)] if rolling else payload
        content = " ".join(c for c in (_clean(line) for line in lines) if c)
        if not content or end <= start:
            continue
        if cues and cues[-1].text == content and start - cues[-1].end < 0.05:
            cues[-1].end = end  # the same caption continued
            continue
        cues.append(Cue(start, end, content))
    return cues


def _timed_words(cues: Sequence[Cue]) -> list[tuple[float, str]]:
    """Caption words with their estimated midpoints (spread evenly over each cue)."""
    out = []
    for cue in cues:
        words = cue.text.split()
        step = (cue.end - cue.start) / max(1, len(words))
        out += [(cue.start + step * (i + 0.5), w) for i, w in enumerate(words)]
    return out


def same_language(caption_language: str | None, transcript_language: str | None) -> bool:
    if not caption_language or not transcript_language:
        return False
    return caption_language.split("-")[0].lower() == transcript_language.split("-")[0].lower()


def compare(segments: Sequence[Any], cues: Sequence[Cue], *, kind: str, language: str | None) -> dict[int, dict[str, Any]]:
    """``{segment index: captions_differ flag}`` for passages whose words
    clearly differ from the captions' words at the same time."""
    words = _timed_words(cues)
    flags: dict[int, dict[str, Any]] = {}
    for index, seg in enumerate(segments):
        text = seg.text if hasattr(seg, "text") else seg["text"]
        start = seg.start if hasattr(seg, "start") else seg["start"]
        end = seg.end if hasattr(seg, "end") else seg["end"]
        ours = [t for t in (normalize_token(w) for w in text.split()) if t]
        theirs_raw = [w for t, w in words if start - SLACK_SECONDS <= t <= end + SLACK_SECONDS]
        theirs = [t for t in (normalize_token(w) for w in theirs_raw) if t]
        if len(ours) < MIN_WORDS or len(theirs) < MIN_WORDS:
            continue  # nothing to compare: captions often leave out short or overlapping speech
        agreement = difflib.SequenceMatcher(None, ours, theirs, autojunk=False).ratio()
        if agreement < AGREEMENT:
            flags[index] = {"type": "captions_differ", "severity": "low", "source": kind, "language": language,
                            "caption_text": " ".join(theirs_raw), "agreement": round(agreement, 2)}
    return flags
