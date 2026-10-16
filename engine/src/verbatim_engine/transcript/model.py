"""In-memory transcript used while building and storing a job's result."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TWord:
    start: float
    end: float
    text: str
    probability: float | None = None
    #: "whisper" (recogniser timing), "alignment" (forced alignment) or
    #: "wildcard" (timed by the aligner's wildcard: digits, symbols, ...)
    timing_source: str = "whisper"
    whisper_start: float | None = None
    whisper_end: float | None = None
    alignment_score: float | None = None
    #: diarization speaker index while building; database id once stored
    speaker: int | None = None


@dataclass
class TSegment:
    start: float
    end: float
    text: str
    words: list[TWord] = field(default_factory=list)
    language: str | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    temperature: float | None = None
    chunk_index: int | None = None
    flags: list[dict[str, Any]] = field(default_factory=list)
    speaker: int | None = None
    speaker_confidence: float | None = None

    @classmethod
    def from_merged(cls, seg) -> "TSegment":
        words = [
            TWord(w.start, w.end, w.text, w.probability, "whisper", w.start, w.end)
            for w in (seg.words or [])
        ]
        return cls(
            start=seg.start,
            end=seg.end,
            text=seg.text,
            words=words,
            language=seg.language,
            avg_logprob=seg.avg_logprob,
            no_speech_prob=seg.no_speech_prob,
            compression_ratio=seg.compression_ratio,
            temperature=seg.temperature,
            chunk_index=seg.chunk_index,
            flags=[dict(f) for f in seg.flags],
        )
