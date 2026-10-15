"""Shared fixtures.

Real-model tests need actual Whisper weights; point ``VERBATIM_TEST_MODEL_DIR``
at a faster-whisper (CTranslate2) model folder, or ``VERBATIM_TEST_MLX_MODEL_DIR``
at an MLX model folder. Without them those tests are skipped — never faked.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

CORPUS = Path(__file__).parent / "corpus"
SR = 16_000


@pytest.fixture(scope="session")
def corpus() -> dict:
    return json.loads((CORPUS / "manifest.json").read_text())


@pytest.fixture(scope="session")
def jfk_path() -> Path:
    return CORPUS / "jfk.wav"


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    return d


def model_dir_from_env(var: str) -> Path | None:
    value = os.environ.get(var)
    if not value:
        return None
    path = Path(value)
    return path if path.is_dir() else None


@pytest.fixture(scope="session")
def fw_model_dir() -> Path:
    path = model_dir_from_env("VERBATIM_TEST_MODEL_DIR")
    if path is None:
        pytest.skip("VERBATIM_TEST_MODEL_DIR not set: real faster-whisper weights required")
    return path


@pytest.fixture(scope="session")
def mlx_model_dir() -> Path:
    path = model_dir_from_env("VERBATIM_TEST_MLX_MODEL_DIR")
    if path is None:
        pytest.skip("VERBATIM_TEST_MLX_MODEL_DIR not set: real MLX weights required")
    return path


# ----------------------------------------------------------------------------- audio helpers


def write_wav(path: Path, samples: np.ndarray, rate: int = SR) -> Path:
    data = np.clip(samples, -1.0, 1.0)
    pcm = (data * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def tts_available() -> bool:
    return shutil.which("espeak-ng") is not None or (sys.platform == "darwin" and shutil.which("say") is not None)


def synthesize(text: str, out_dir: Path, name: str, lang: str = "en") -> np.ndarray:
    """Speech for a known text, as 16 kHz float32. Uses espeak-ng or macOS `say`,
    then the engine's own decoder/resampler."""
    from verbatim_engine.audio.extract import extract_audio

    out_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("espeak-ng"):
        raw = out_dir / f"{name}.raw.wav"
        voice = {"en": "en-us", "de": "de", "es": "es", "fr": "fr"}.get(lang, lang)
        subprocess.run(["espeak-ng", "-v", voice, "-s", "150", "-w", str(raw), text], check=True, capture_output=True)
    elif sys.platform == "darwin" and shutil.which("say"):
        raw = out_dir / f"{name}.raw.aiff"
        subprocess.run(["say", "-o", str(raw), text], check=True, capture_output=True)
    else:
        pytest.skip("no speech synthesizer (espeak-ng or say) available")
    wav = out_dir / f"{name}.wav"
    extract_audio(raw, wav)
    return read_wav(wav)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


# ----------------------------------------------------------------------------- media encoding


def encode_media(
    path: Path,
    audio: np.ndarray | None,
    *,
    container: str = "mp4",
    audio_codec: str = "aac",
    video: bool = False,
    video_seconds: float | None = None,
    audio_offset: float = 0.0,
) -> Path:
    """Encode test media with PyAV (no ffmpeg binary needed)."""
    import av

    out = av.open(str(path), mode="w", format=container)
    vstream = None
    astream = None
    if video:
        vstream = out.add_stream("mpeg4", rate=10)
        vstream.width, vstream.height, vstream.pix_fmt = 64, 48, "yuv420p"
    if audio is not None:
        astream = out.add_stream(audio_codec, rate=SR)
        astream.layout = "mono"
    if vstream is not None:
        seconds = video_seconds if video_seconds is not None else (len(audio) / SR if audio is not None else 1.0)
        for i in range(int(seconds * 10)):
            img = np.full((48, 64, 3), (i * 7) % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = i
            for packet in vstream.encode(frame):
                out.mux(packet)
        for packet in vstream.encode(None):
            out.mux(packet)
    if astream is not None:
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        frame_size = astream.codec_context.frame_size or 1024
        offset_samples = int(round(audio_offset * SR))
        for start in range(0, len(pcm), frame_size):
            block = pcm[start : start + frame_size]
            if block.size < frame_size:
                block = np.pad(block, (0, frame_size - block.size))
            frame = av.AudioFrame.from_ndarray(block.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = SR
            frame.pts = start + offset_samples
            for packet in astream.encode(frame):
                out.mux(packet)
        for packet in astream.encode(None):
            out.mux(packet)
    out.close()
    return path
