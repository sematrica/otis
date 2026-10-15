"""Split long recordings into chunks at quiet points.

Each chunk *owns* the interval between its two cut points.

* **Cut in a pause** (the usual case): the audio given to each chunk extends
  into the pause but never past it, so neighbouring chunks never hear the same
  speech and their results are simply concatenated.
* **Cut inside continuous speech** (no pause found near the boundary): the
  audio is padded by ``pad_seconds`` on both sides so the words around the
  cut are decoded twice, and the merger splices the two decodings where they
  agree (see :mod:`verbatim_engine.asr.merge`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from ..errors import Cancelled
from . import SAMPLE_RATE
from .wav import WavReader

FRAME_SECONDS = 0.02
_FRAME_SAMPLES = int(FRAME_SECONDS * SAMPLE_RATE)
_SILENCE_FLOOR_DB = -100.0
#: Cut energy above this suggests no real pause was found near the boundary.
SPEECH_LEVEL_DB = -35.0


@dataclass(frozen=True)
class Chunk:
    index: int
    owned_start: float
    owned_end: float
    audio_start: float
    audio_end: float
    #: Smoothed energy (dBFS) at the cut that starts this chunk; None for the first.
    cut_energy_db: float | None = None

    @property
    def cut_in_speech(self) -> bool:
        return self.cut_energy_db is not None and self.cut_energy_db > SPEECH_LEVEL_DB

    def to_dict(self) -> dict:
        data = asdict(self)
        data["cut_in_speech"] = self.cut_in_speech
        return data


def energy_envelope(
    reader: WavReader,
    progress: Callable[[float, float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Per-20 ms RMS level in dBFS, computed in one streaming pass."""
    block = _FRAME_SAMPLES * 500  # 10 s
    levels: list[np.ndarray] = []
    done = 0
    for data in reader.iter_blocks(block):
        if should_cancel is not None and should_cancel():
            raise Cancelled("Cancelled during audio analysis.")
        usable = (data.size // _FRAME_SAMPLES) * _FRAME_SAMPLES
        frames = data[:usable].astype(np.float32).reshape(-1, _FRAME_SAMPLES) / 32768.0
        if data.size > usable:  # trailing partial frame
            tail = data[usable:].astype(np.float32) / 32768.0
            frames = np.vstack([frames, np.pad(tail, (0, _FRAME_SAMPLES - tail.size))]) if frames.size else tail[None, :]
        rms = np.sqrt(np.mean(frames * frames, axis=1))
        levels.append(np.maximum(20.0 * np.log10(rms + 1e-12), _SILENCE_FLOOR_DB).astype(np.float32))
        done += data.size
        if progress is not None:
            progress(done / SAMPLE_RATE, reader.duration)
    return np.concatenate(levels) if levels else np.zeros(0, dtype=np.float32)


def _quiet_extent(envelope: np.ndarray, cut: float, level_db: float) -> tuple[float, float]:
    """The contiguous quiet region around ``cut`` (seconds)."""
    threshold = min(SPEECH_LEVEL_DB, level_db + 15.0)
    centre = min(max(int(cut / FRAME_SECONDS), 0), envelope.size - 1)
    lo = centre
    while lo > 0 and envelope[lo - 1] <= threshold:
        lo -= 1
    hi = centre
    while hi < envelope.size - 1 and envelope[hi + 1] <= threshold:
        hi += 1
    return lo * FRAME_SECONDS, (hi + 1) * FRAME_SECONDS


def _best_cut(envelope: np.ndarray, lo: float, hi: float, target: float, smooth_frames: int) -> tuple[float, float]:
    """Return (cut_time, smoothed_db) for the quietest point in [lo, hi]."""
    first = max(0, int(lo / FRAME_SECONDS))
    last = min(envelope.size, int(hi / FRAME_SECONDS) + 1)
    region = envelope[first:last]
    if region.size < smooth_frames:
        return target, float(np.mean(region)) if region.size else _SILENCE_FLOOR_DB
    # Average in the power domain so one quiet frame inside speech is not a "pause".
    power = np.power(10.0, region / 10.0)
    smoothed = np.convolve(power, np.ones(smooth_frames) / smooth_frames, mode="valid")
    smoothed_db = 10.0 * np.log10(smoothed + 1e-20)
    floor = float(smoothed_db.min())
    candidates = np.flatnonzero(smoothed_db <= floor + 1.0)
    centres = (first + candidates + smooth_frames / 2.0) * FRAME_SECONDS
    best = int(np.argmin(np.abs(centres - target)))
    return float(centres[best]), float(smoothed_db[candidates[best]])


def plan_chunks(
    duration: float,
    envelope: np.ndarray | None,
    *,
    target_seconds: float = 600.0,
    search_seconds: float = 30.0,
    pad_seconds: float = 1.0,
    single_chunk_max: float | None = None,
    smooth_seconds: float = 0.5,
) -> list[Chunk]:
    if duration <= 0:
        return []
    if single_chunk_max is None:
        single_chunk_max = target_seconds * 1.2
    if duration <= single_chunk_max or envelope is None or envelope.size == 0:
        return [Chunk(0, 0.0, duration, 0.0, duration, None)]

    smooth_frames = max(1, int(smooth_seconds / FRAME_SECONDS))
    cuts: list[tuple[float, float]] = []
    position = 0.0
    while duration - position > single_chunk_max:
        target = position + target_seconds
        lo = max(position + target_seconds / 2.0, target - search_seconds)
        hi = min(duration - target_seconds / 4.0, target + search_seconds)
        cut, level = _best_cut(envelope, lo, hi, target, smooth_frames)
        if cut <= position:  # defensive: never produce an empty chunk
            cut = target
        cuts.append((cut, level))
        position = cut

    # For each cut: how far the left chunk's audio may extend past it, and how
    # far before it the right chunk's audio may start.
    reach_after: list[float] = []
    reach_before: list[float] = []
    for cut, level in cuts:
        if level > SPEECH_LEVEL_DB:
            reach_after.append(min(duration, cut + pad_seconds))
            reach_before.append(max(0.0, cut - pad_seconds))
        else:
            quiet_start, quiet_end = _quiet_extent(envelope, cut, level)
            reach_after.append(min(duration, cut + pad_seconds, max(cut, quiet_end)))
            reach_before.append(max(0.0, cut - pad_seconds, min(cut, quiet_start)))

    bounds = [0.0] + [c for c, _ in cuts] + [duration]
    levels: list[float | None] = [None] + [lvl for _, lvl in cuts]
    chunks = []
    for i in range(len(bounds) - 1):
        chunks.append(
            Chunk(
                index=i,
                owned_start=bounds[i],
                owned_end=bounds[i + 1],
                audio_start=reach_before[i - 1] if i > 0 else 0.0,
                audio_end=reach_after[i] if i < len(cuts) else duration,
                cut_energy_db=levels[i],
            )
        )
    return chunks
