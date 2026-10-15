"""mlx-whisper backend — Metal GPU acceleration on Apple Silicon.

Differences from faster-whisper that the job records honestly:

* no beam search (mlx-whisper raises NotImplementedError): greedy decoding with
  temperature fallback is used regardless of the requested beam size;
* no built-in VAD: the engine runs Silero VAD itself and passes the speech
  regions as ``clip_timestamps`` so silence is never decoded;
* language is detected once per chunk, on the first 30 s of detected speech.

``mlx_whisper.load_models.load_model`` silently downloads from HuggingFace when
the given path does not exist, so the directory is validated before loading
and workers run with ``HF_HUB_OFFLINE=1``.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import logging
import threading
import types
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..audio import SAMPLE_RATE
from ..errors import BackendUnavailable, ModelNotInstalled
from ..util.hardware import is_apple_silicon, macos_version
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
from .vad import speech_regions

log = logging.getLogger(__name__)
_PATCH_LOCK = threading.Lock()


@contextlib.contextmanager
def _progress_reporting(callback: ProgressFn | None) -> Iterator[None]:
    """mlx-whisper reports progress only through a tqdm bar over mel frames.
    Swap in a minimal bar that forwards the real frame counts."""
    if callback is None:
        yield
        return
    module = importlib.import_module("mlx_whisper.transcribe")

    class _Bar:
        def __init__(self, total=None, **_kwargs):
            self.total = total or 0
            self.n = 0

        def update(self, n=1):
            self.n += n
            if self.total:
                callback(min(1.0, self.n / self.total))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    with _PATCH_LOCK:
        original = module.tqdm
        module.tqdm = types.SimpleNamespace(tqdm=_Bar)
        try:
            yield
        finally:
            module.tqdm = original


class MlxWhisperBackend(Backend):
    name = "mlx"
    display_name = "MLX (Apple GPU)"
    required_files = (("config.json",), ("weights.safetensors", "weights.npz"))

    def __init__(self) -> None:
        self._model = None
        self._model_path: str | None = None
        self._dtype_name = "float16"

    @classmethod
    def availability(cls) -> tuple[bool, str | None]:
        if not is_apple_silicon():
            return False, "MLX requires a Mac with Apple Silicon."
        version = macos_version()
        if version is not None and version < (14,):
            return False, "The bundled MLX build requires macOS 14 or newer."
        for module in ("mlx", "mlx_whisper"):
            if importlib.util.find_spec(module) is None:
                return False, f"Python package '{module}' is not installed."
        return True, None

    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            beam_search=False,
            word_timestamps=True,
            builtin_vad=False,
            language_detection=True,
            device="gpu",
            acceleration="MLX on Metal",
        )

    def load(self, model_dir: Path, options: LoadOptions) -> dict[str, Any]:
        available, reason = self.availability()
        if not available:
            raise BackendUnavailable(reason or "MLX is unavailable")
        if not model_dir.is_dir():
            raise ModelNotInstalled("The model folder does not exist.", data={"path": str(model_dir)})
        missing = self.missing_files(model_dir)
        if missing:
            raise ModelNotInstalled(
                f"The model folder is incomplete (missing {', '.join(missing)}).",
                data={"path": str(model_dir), "missing": missing},
            )
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder

        try:
            metal_ok = bool(mx.metal.is_available())
        except AttributeError:  # very old/new MLX without this helper
            metal_ok = True
        if not metal_ok:
            raise BackendUnavailable("Metal is not available on this Mac.")

        mx.random.seed(0)  # temperature fallback samples; keep results reproducible
        self._model_path = str(model_dir)
        # ModelHolder caches the model; mlx_whisper.transcribe reuses it as long
        # as the same path and dtype are passed.
        self._model = ModelHolder.get_model(self._model_path, mx.float16)
        return {"device": "gpu", "dtype": self._dtype_name, "random_seed": 0}

    def _detect_language(self, audio: np.ndarray, start: float) -> tuple[str, float | None]:
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim

        if not self._model.is_multilingual:
            return "en", None
        first = int(start * SAMPLE_RATE)
        window = audio[first : first + N_SAMPLES]
        mel = log_mel_spectrogram(window, n_mels=self._model.dims.n_mels, padding=N_SAMPLES)
        mel_segment = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
        _, probs = self._model.detect_language(mel_segment)
        language = max(probs, key=probs.get)
        return language, float(probs[language])

    def transcribe(
        self, audio: np.ndarray, options: DecodeOptions, progress: ProgressFn | None = None
    ) -> tuple[TranscriptionResult, dict[str, Any]]:
        if self._model is None or self._model_path is None:
            raise RuntimeError("model not loaded")
        import mlx_whisper

        duration = audio.shape[0] / SAMPLE_RATE
        effective: dict[str, Any] = {
            "backend": self.name,
            "device": "gpu",
            "dtype": self._dtype_name,
            "beam_size": 1,
            "beam_search": "not supported by mlx-whisper; greedy decoding used",
            "best_of": max(1, int(options.best_of)),
            "temperatures": list(options.temperatures),
            "compression_ratio_threshold": options.compression_ratio_threshold,
            "log_prob_threshold": options.log_prob_threshold,
            "no_speech_threshold": options.no_speech_threshold,
            "condition_on_previous_text": options.condition_on_previous_text,
            "word_timestamps": options.word_timestamps,
            "vad": "silero (engine)" if options.vad else "off",
            "hallucination_silence_threshold": options.hallucination_silence_threshold,
        }

        clip: list[float] | str = "0"
        first_speech = 0.0
        if options.vad:
            regions = speech_regions(audio)
            if not regions:
                if progress is not None:
                    progress(1.0)
                return (
                    TranscriptionResult(
                        language=options.language,
                        language_probability=None,
                        language_source="user" if options.language else "detected",
                        duration=duration,
                        segments=[],
                    ),
                    effective,
                )
            clip = [t for region in regions for t in region]
            first_speech = regions[0][0]

        if options.language:
            language, probability, source = options.language, None, "user"
        else:
            language, probability = self._detect_language(audio, first_speech)
            source = "detected"

        kwargs: dict[str, Any] = {}
        if max(1, int(options.best_of)) > 1:
            kwargs["best_of"] = int(options.best_of)
        with _progress_reporting(progress):
            raw = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self._model_path,
                verbose=None,
                temperature=tuple(options.temperatures),
                compression_ratio_threshold=options.compression_ratio_threshold,
                logprob_threshold=options.log_prob_threshold,
                no_speech_threshold=options.no_speech_threshold,
                condition_on_previous_text=options.condition_on_previous_text,
                initial_prompt=options.initial_prompt or None,
                word_timestamps=options.word_timestamps,
                clip_timestamps=clip,
                hallucination_silence_threshold=options.hallucination_silence_threshold,
                language=language,
                task="transcribe",
                fp16=True,
                **kwargs,
            )
        if progress is not None:
            progress(1.0)

        segments: list[Segment] = []
        for seg in raw.get("segments", []):
            words = None
            if options.word_timestamps and seg.get("words") is not None:
                words = [
                    Word(float(w["start"]), float(w["end"]), w["word"], float(w.get("probability", 0.0)))
                    for w in seg["words"]
                ]
            segments.append(
                Segment(
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=seg["text"],
                    words=words,
                    avg_logprob=_maybe_float(seg.get("avg_logprob")),
                    no_speech_prob=_maybe_float(seg.get("no_speech_prob")),
                    compression_ratio=_maybe_float(seg.get("compression_ratio")),
                    temperature=_maybe_float(seg.get("temperature")),
                )
            )
        result = TranscriptionResult(
            language=raw.get("language", language),
            language_probability=probability,
            language_source=source,
            duration=duration,
            segments=segments,
        )
        return result, effective

    def close(self) -> None:
        self._model = None


def _maybe_float(value: Any) -> float | None:
    return float(value) if value is not None else None
