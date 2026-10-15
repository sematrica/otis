"""Speech Recognition Engine: backend-independent types and interface.

A backend turns 16 kHz mono float32 audio into timestamped segments. Times in a
:class:`TranscriptionResult` are relative to the start of the audio it was given.

Backends must not download anything: they receive a local model directory
that the Model Manager has already validated.
"""

from __future__ import annotations

import abc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar

import numpy as np

ProgressFn = Callable[[float], None]
"""Receives the fraction (0–1) of the given audio processed so far."""


@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: float | None = None


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    temperature: float | None = None

    def shifted(self, offset: float) -> "Segment":
        return Segment(
            start=self.start + offset,
            end=self.end + offset,
            text=self.text,
            words=[Word(w.start + offset, w.end + offset, w.text, w.probability) for w in self.words]
            if self.words is not None
            else None,
            avg_logprob=self.avg_logprob,
            no_speech_prob=self.no_speech_prob,
            compression_ratio=self.compression_ratio,
            temperature=self.temperature,
        )


def shift_segments(segments: list[Segment], offset: float) -> list[Segment]:
    """Shift times by ``offset`` seconds, clamping at 0 (used to undo a lead-in)."""
    shifted = []
    for seg in segments:
        moved = seg.shifted(offset)
        moved.start, moved.end = max(0.0, moved.start), max(0.0, moved.end)
        if moved.words:
            for w in moved.words:
                w.start, w.end = max(0.0, w.start), max(0.0, w.end)
        shifted.append(moved)
    return shifted


@dataclass
class TranscriptionResult:
    language: str | None
    language_probability: float | None
    #: "detected" or "user" (forced by the caller)
    language_source: str
    duration: float
    segments: list[Segment] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptionResult":
        segments = []
        for s in data.get("segments", []):
            s = dict(s)
            words = s.pop("words", None)
            segments.append(Segment(**s, words=[Word(**w) for w in words] if words is not None else None))
        return cls(
            language=data.get("language"),
            language_probability=data.get("language_probability"),
            language_source=data.get("language_source", "detected"),
            duration=data.get("duration", 0.0),
            segments=segments,
        )


@dataclass(frozen=True)
class Capabilities:
    beam_search: bool
    word_timestamps: bool
    builtin_vad: bool
    language_detection: bool
    device: str
    acceleration: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DecodeOptions:
    """Decoding settings. The task is always transcription in the spoken
    language; translation is deliberately not exposed."""

    language: str | None = None
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    temperatures: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    compression_ratio_threshold: float | None = 2.4
    log_prob_threshold: float | None = -1.0
    no_speech_threshold: float | None = 0.6
    condition_on_previous_text: bool = True
    initial_prompt: str | None = None
    word_timestamps: bool = True
    vad: bool = True
    hallucination_silence_threshold: float | None = None


@dataclass
class LoadOptions:
    #: CTranslate2 compute type for CPU backends ("int8", "float32", ...).
    compute_type: str = "int8"
    #: 0 = choose automatically (performance-core count).
    cpu_threads: int = 0


class Backend(abc.ABC):
    name: ClassVar[str]
    display_name: ClassVar[str]
    #: Files that must exist in a model directory for this backend.
    required_files: ClassVar[tuple[tuple[str, ...], ...]]

    @classmethod
    @abc.abstractmethod
    def availability(cls) -> tuple[bool, str | None]:
        """Cheap check (no heavy imports) of whether this backend can run here."""

    @classmethod
    @abc.abstractmethod
    def capabilities(cls) -> Capabilities: ...

    @classmethod
    def missing_files(cls, model_dir: Path) -> list[str]:
        """Each entry of ``required_files`` is a group of alternatives."""
        missing = []
        for alternatives in cls.required_files:
            if not any((model_dir / name).is_file() for name in alternatives):
                missing.append(" or ".join(alternatives))
        return missing

    @abc.abstractmethod
    def load(self, model_dir: Path, options: LoadOptions) -> dict[str, Any]:
        """Load the model; returns the effective load settings."""

    @abc.abstractmethod
    def transcribe(
        self, audio: np.ndarray, options: DecodeOptions, progress: ProgressFn | None = None
    ) -> tuple[TranscriptionResult, dict[str, Any]]:
        """Returns the result and the *effective* decoding settings used."""

    def close(self) -> None:  # pragma: no cover - default no-op
        pass
