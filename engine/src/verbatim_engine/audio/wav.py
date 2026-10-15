"""Random-access reader for the engine's 16 kHz mono 16-bit WAV files.

Only the requested range is read into memory.
"""

from __future__ import annotations

import os
import wave
from typing import Iterator

import numpy as np

from ..errors import MediaError
from . import SAMPLE_RATE


class WavReader:
    def __init__(self, path: str | os.PathLike[str]):
        self.path = str(path)
        try:
            self._wav = wave.open(self.path, "rb")
        except (wave.Error, EOFError, OSError) as exc:
            raise MediaError(f"Extracted audio is unreadable: {exc}") from exc
        if (
            self._wav.getnchannels() != 1
            or self._wav.getsampwidth() != 2
            or self._wav.getframerate() != SAMPLE_RATE
        ):
            self._wav.close()
            raise MediaError("Extracted audio has an unexpected format.")
        self.samples = self._wav.getnframes()

    @property
    def duration(self) -> float:
        return self.samples / SAMPLE_RATE

    def read_int16(self, start_sample: int, count: int) -> np.ndarray:
        start_sample = max(0, min(start_sample, self.samples))
        count = max(0, min(count, self.samples - start_sample))
        self._wav.setpos(start_sample)
        return np.frombuffer(self._wav.readframes(count), dtype="<i2")

    def read(self, start: float, end: float) -> np.ndarray:
        """float32 samples in [-1, 1) for the time range [start, end)."""
        first = int(round(start * SAMPLE_RATE))
        last = int(round(end * SAMPLE_RATE))
        data = self.read_int16(first, last - first)
        return data.astype(np.float32) / 32768.0

    def iter_blocks(self, block_samples: int) -> Iterator[np.ndarray]:
        position = 0
        while position < self.samples:
            block = self.read_int16(position, block_samples)
            if block.size == 0:
                break
            yield block
            position += block.size

    def close(self) -> None:
        self._wav.close()

    def __enter__(self) -> "WavReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
