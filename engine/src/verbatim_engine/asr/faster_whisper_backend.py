"""faster-whisper (CTranslate2) backend — runs on the CPU everywhere.

Supports beam search, temperature fallback, built-in Silero VAD and
word timestamps (DTW over the model's alignment heads).
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..audio import SAMPLE_RATE
from ..errors import BackendUnavailable, ModelNotInstalled
from ..util.hardware import performance_core_count
from .base import (
    Backend,
    Capabilities,
    DecodeOptions,
    LoadOptions,
    ProgressFn,
    Segment,
    TranscriptionResult,
    Word,
)

log = logging.getLogger(__name__)


class FasterWhisperBackend(Backend):
    name = "faster-whisper"
    display_name = "faster-whisper (CPU)"
    # tokenizer.json is required: without it faster-whisper would try to fetch
    # one from the network.
    required_files = (("model.bin",), ("config.json",), ("tokenizer.json",))

    def __init__(self) -> None:
        self._model = None
        self._load_settings: dict[str, Any] = {}

    @classmethod
    def availability(cls) -> tuple[bool, str | None]:
        for module in ("faster_whisper", "ctranslate2"):
            if importlib.util.find_spec(module) is None:
                return False, f"Python package '{module}' is not installed."
        return True, None

    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            beam_search=True,
            word_timestamps=True,
            builtin_vad=True,
            language_detection=True,
            device="cpu",
            acceleration="CTranslate2 on CPU",
        )

    def load(self, model_dir: Path, options: LoadOptions) -> dict[str, Any]:
        available, reason = self.availability()
        if not available:
            raise BackendUnavailable(reason or "faster-whisper is unavailable")
        missing = self.missing_files(model_dir)
        if missing:
            raise ModelNotInstalled(
                f"The model folder is incomplete (missing {', '.join(missing)}).",
                data={"path": str(model_dir), "missing": missing},
            )
        from faster_whisper import WhisperModel

        threads = options.cpu_threads or performance_core_count()
        self._model = WhisperModel(
            str(model_dir),
            device="cpu",
            compute_type=options.compute_type,
            cpu_threads=threads,
            num_workers=1,
            local_files_only=True,
        )
        self._load_settings = {"device": "cpu", "compute_type": options.compute_type, "cpu_threads": threads}
        return dict(self._load_settings)

    def transcribe(
        self, audio: np.ndarray, options: DecodeOptions, progress: ProgressFn | None = None
    ) -> tuple[TranscriptionResult, dict[str, Any]]:
        if self._model is None:
            raise RuntimeError("model not loaded")
        duration = audio.shape[0] / SAMPLE_RATE
        beam_size = max(1, int(options.beam_size))
        best_of = max(1, int(options.best_of))
        segments_iter, info = self._model.transcribe(
            audio,
            language=options.language,
            task="transcribe",
            beam_size=beam_size,
            best_of=best_of,
            patience=options.patience,
            temperature=list(options.temperatures),
            compression_ratio_threshold=options.compression_ratio_threshold,
            log_prob_threshold=options.log_prob_threshold,
            no_speech_threshold=options.no_speech_threshold,
            condition_on_previous_text=options.condition_on_previous_text,
            prompt_reset_on_temperature=0.5,
            initial_prompt=options.initial_prompt or None,
            word_timestamps=options.word_timestamps,
            vad_filter=options.vad,
            hallucination_silence_threshold=options.hallucination_silence_threshold,
            language_detection_segments=4,
        )
        segments: list[Segment] = []
        for seg in segments_iter:  # decoding happens lazily while iterating
            words = None
            if options.word_timestamps and seg.words is not None:
                words = [Word(float(w.start), float(w.end), w.word, float(w.probability)) for w in seg.words]
            segments.append(
                Segment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=seg.text,
                    words=words,
                    avg_logprob=float(seg.avg_logprob),
                    no_speech_prob=float(seg.no_speech_prob),
                    compression_ratio=float(seg.compression_ratio),
                    temperature=float(seg.temperature) if seg.temperature is not None else None,
                )
            )
            if progress is not None and duration > 0:
                progress(min(1.0, float(seg.end) / duration))
        if progress is not None:
            progress(1.0)

        has_speech = bool(segments)
        result = TranscriptionResult(
            language=info.language if has_speech or options.language else None,
            language_probability=float(info.language_probability) if options.language is None and has_speech else None,
            language_source="user" if options.language else "detected",
            duration=duration,
            segments=segments,
        )
        effective = {
            "backend": self.name,
            **self._load_settings,
            "beam_size": beam_size,
            "best_of": best_of,
            "patience": options.patience,
            "temperatures": list(options.temperatures),
            "compression_ratio_threshold": options.compression_ratio_threshold,
            "log_prob_threshold": options.log_prob_threshold,
            "no_speech_threshold": options.no_speech_threshold,
            "condition_on_previous_text": options.condition_on_previous_text,
            "word_timestamps": options.word_timestamps,
            "vad": "silero (built-in)" if options.vad else "off",
            "hallucination_silence_threshold": options.hallucination_silence_threshold,
        }
        return result, effective

    def close(self) -> None:
        self._model = None
