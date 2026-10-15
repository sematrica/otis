"""Voice activity detection with Silero VAD (the ONNX model bundled inside the
faster-whisper package, so nothing is downloaded)."""

from __future__ import annotations

import numpy as np

from ..audio import SAMPLE_RATE


def speech_regions(
    audio: np.ndarray,
    *,
    min_silence_ms: int = 2000,
    speech_pad_ms: int = 400,
    threshold: float = 0.5,
) -> list[tuple[float, float]]:
    """Speech regions in seconds, relative to the start of ``audio``."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(threshold=threshold, min_silence_duration_ms=min_silence_ms, speech_pad_ms=speech_pad_ms)
    stamps = get_speech_timestamps(audio, options, sampling_rate=SAMPLE_RATE)
    return [(s["start"] / SAMPLE_RATE, s["end"] / SAMPLE_RATE) for s in stamps]
