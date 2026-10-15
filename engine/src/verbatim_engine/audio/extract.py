"""Decode any supported media to 16 kHz mono 16-bit PCM WAV, streaming.

The output timeline matches the media's presentation timeline, so transcript
timestamps line up with video playback:

* time 0 of the WAV is the container's start time (what players show as 0:00);
* a leading offset (audio starting after the video) is filled with silence;
* gaps between audio frames are filled with silence, overlaps are trimmed;
* timestamp discontinuities that cannot be real (large jumps, e.g. a broken
  concatenated stream) are *not* "corrected" — decoding re-anchors and carries
  on, so no audio is ever discarded because of bad timestamps.

Memory use is bounded by one decoded frame at a time.
"""

from __future__ import annotations

import logging
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..errors import Cancelled, MediaError, NoAudioStream
from . import SAMPLE_RATE

log = logging.getLogger(__name__)

#: Timing errors below this are treated as jitter, not gaps.
GAP_TOLERANCE_S = 0.05
#: Largest gap filled with silence; larger forward jumps are discontinuities.
MAX_GAP_FILL_S = 30.0
#: Largest overlap trimmed; larger backward jumps are discontinuities.
MAX_OVERLAP_TRIM_S = 1.0
#: WAV uses 32-bit chunk sizes; stay safely below 4 GiB (~37 hours at 16 kHz).
MAX_WAV_BYTES = 0xFFFFFFFF - 1024 * 1024

ProgressFn = Callable[[float, float | None], None]


@dataclass
class ExtractResult:
    path: Path
    sample_rate: int
    samples: int
    duration: float
    decode_errors: int
    inserted_silence: float
    trimmed_overlap: float
    discontinuities: int


class _WavSink:
    def __init__(self, path: Path):
        self.path = path
        self._wav = wave.open(str(path), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(SAMPLE_RATE)
        self.samples = 0

    def write(self, data: np.ndarray) -> None:
        if data.size == 0:
            return
        if (self.samples + data.size) * 2 > MAX_WAV_BYTES:
            raise MediaError("Recordings longer than about 37 hours are not supported.")
        self._wav.writeframesraw(np.ascontiguousarray(data, dtype="<i2").tobytes())
        self.samples += data.size

    def write_silence(self, samples: int) -> None:
        block = 16 * SAMPLE_RATE
        zeros = np.zeros(min(block, max(samples, 0)), dtype=np.int16)
        remaining = samples
        while remaining > 0:
            n = min(remaining, block)
            self.write(zeros[:n])
            remaining -= n

    def close(self) -> None:
        self._wav.close()


def extract_audio(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    stream_index: int | None = None,
    expected_duration: float | None = None,
    progress: ProgressFn | None = None,
    should_cancel: Callable[[], bool] | None = None,
    max_decode_errors: int = 100,
) -> ExtractResult:
    import av

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")

    try:
        container = av.open(str(source), mode="r", metadata_errors="ignore")
    except av.error.FFmpegError as exc:
        raise MediaError("The file could not be opened for decoding.", data={"detail": str(exc)}) from exc

    sink: _WavSink | None = None
    try:
        if stream_index is not None:
            matches = [s for s in container.streams.audio if s.index == stream_index]
            if not matches:
                raise NoAudioStream(f"Audio stream {stream_index} not found.")
            stream = matches[0]
        elif container.streams.audio:
            stream = container.streams.audio[0]
        else:
            raise NoAudioStream("The file contains no audio stream.")

        reference = container.start_time / 1_000_000 if container.start_time is not None else None
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        sink = _WavSink(partial)

        decode_errors = 0
        inserted = 0.0
        trimmed = 0.0
        discontinuities = 0
        pending_trim = 0  # output samples still to drop because of an overlap
        last_report = -1.0

        def emit(frames) -> None:
            nonlocal pending_trim, trimmed
            for out in frames:
                data = out.to_ndarray().reshape(-1)
                if pending_trim:
                    cut = min(pending_trim, data.size)
                    data = data[cut:]
                    pending_trim -= cut
                    trimmed += cut / SAMPLE_RATE
                sink.write(data)

        for packet in container.demux(stream):
            if should_cancel is not None and should_cancel():
                raise Cancelled("Audio extraction was cancelled.")
            try:
                frames = packet.decode()
            except av.error.FFmpegError as exc:
                decode_errors += 1
                log.warning("decode error %d in %s: %s", decode_errors, source, exc)
                if decode_errors > max_decode_errors:
                    raise MediaError(
                        "Too many decoding errors; the file appears to be damaged.",
                        data={"decode_errors": decode_errors},
                    ) from exc
                continue

            for frame in frames:
                t = frame.time
                if t is not None:
                    if reference is None:
                        reference = t
                    rel = t - reference
                    # Where the next sample will land once pending trims are applied.
                    gap = rel - (sink.samples - pending_trim) / SAMPLE_RATE
                    if GAP_TOLERANCE_S < gap <= MAX_GAP_FILL_S:
                        sink.write_silence(int(round(gap * SAMPLE_RATE)))
                        inserted += gap
                    elif -MAX_OVERLAP_TRIM_S <= gap < -GAP_TOLERANCE_S:
                        pending_trim += int(round(-gap * SAMPLE_RATE))
                    elif abs(gap) > GAP_TOLERANCE_S:
                        # Implausible jump: keep the audio contiguous and re-anchor.
                        discontinuities += 1
                        reference = t - sink.samples / SAMPLE_RATE
                        log.warning("timestamp discontinuity of %.3fs at %.3fs in %s", gap, rel, source)
                emit(resampler.resample(frame))

                if progress is not None:
                    done = sink.samples / SAMPLE_RATE
                    if done - last_report >= 1.0:
                        last_report = done
                        progress(done, expected_duration)

        emit(resampler.resample(None))

        if sink.samples == 0:
            raise MediaError("No audio could be decoded from the file.", data={"decode_errors": decode_errors})

        sink.close()
        sink = None
        os.replace(partial, destination)
        duration = _wav_samples(destination) / SAMPLE_RATE
        if progress is not None:
            progress(duration, expected_duration if expected_duration else duration)
        return ExtractResult(
            path=destination,
            sample_rate=SAMPLE_RATE,
            samples=int(round(duration * SAMPLE_RATE)),
            duration=duration,
            decode_errors=decode_errors,
            inserted_silence=inserted,
            trimmed_overlap=trimmed,
            discontinuities=discontinuities,
        )
    except BaseException:
        if sink is not None:
            try:
                sink.close()
            except Exception:  # noqa: BLE001 - best effort cleanup
                pass
        partial.unlink(missing_ok=True)
        raise
    finally:
        container.close()


def _wav_samples(path: Path) -> int:
    with wave.open(str(path), "rb") as w:
        return w.getnframes()
