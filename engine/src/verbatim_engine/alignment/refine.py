"""Turn CTC token positions into word boundaries.

A CTC model marks each token with a short "spike", usually inside the sound
it stands for, not at its edges: measured on speech with known word timings
(``scripts/benchmark_word_timing.py``), the first spike of a word comes
about 60 ms after the word starts in connected speech and the last spike
about 110 ms before it ends. Between two words the true boundary lies in the
gap between their spikes, so:

* if the gap contains a pause (audio energy near the recording's quiet
  level for at least ``MIN_PAUSE_SECONDS``), the first word ends where the
  sound stops and the next starts where it resumes;
* otherwise (connected speech) the gap is split at ``CONNECTED_SPLIT`` of its
  length from the first word.

Group edges use the energy alone: the first word may start earlier than its
first spike if the sound is already there, and the last word ends where the
sound stops (both bounded).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..audio import SAMPLE_RATE

HOP_SECONDS = 0.01
MIN_PAUSE_SECONDS = 0.06
#: Fraction of an inter-word gap (between spikes) that belongs to the first
#: word, for connected speech. Fitted on synthetic speech (see module doc).
CONNECTED_SPLIT = 0.62
MAX_START_EXTENSION = 0.15
MAX_END_EXTENSION = 0.40
#: Energy above the quiet level (dB) that counts as sound.
SOUND_MARGIN_DB = 12.0


@dataclass
class Span:
    start: float
    end: float


def _levels(audio: np.ndarray) -> np.ndarray:
    hop = int(HOP_SECONDS * SAMPLE_RATE)
    n = audio.size // hop
    if n == 0:
        return np.zeros(0)
    frames = audio[: n * hop].reshape(n, hop).astype(np.float64)
    return 10 * np.log10((frames**2).mean(axis=1) + 1e-12)


def refine(spans: list[Span], audio: np.ndarray, offset: float) -> list[Span]:
    """Adjusted copies of ``spans`` (absolute seconds) for audio starting at ``offset``."""
    if not spans:
        return []
    levels = _levels(audio)
    if levels.size == 0:
        return [Span(s.start, s.end) for s in spans]
    quiet = float(np.percentile(levels, 10))
    loud = float(np.percentile(levels, 95))
    if loud - quiet < SOUND_MARGIN_DB:
        # no usable contrast (constant noise, music, silence): energy cannot
        # tell pauses from speech, so only the connected-speech rule applies
        sound = np.zeros(levels.size, dtype=bool)
        pauses_detectable = False
    else:
        sound = levels > min(quiet + SOUND_MARGIN_DB, loud - 6.0)
        pauses_detectable = True

    def index(t: float) -> int:
        return int(np.clip(np.floor((t - offset) / HOP_SECONDS), 0, levels.size))

    def time(i: int) -> float:
        return offset + i * HOP_SECONDS

    out = [Span(s.start, s.end) for s in spans]
    min_pause = int(round(MIN_PAUSE_SECONDS / HOP_SECONDS))
    for k in range(len(out) - 1):
        a, b = out[k], out[k + 1]
        gap0, gap1 = a.end, spans[k + 1].start
        if gap1 <= gap0:
            continue
        i0, i1 = index(gap0), index(gap1)
        quiet_run = _longest_quiet_run(sound[i0:i1]) if pauses_detectable else None
        if quiet_run is not None and quiet_run[1] - quiet_run[0] >= min_pause:
            a.end = max(gap0, min(gap1, time(i0 + quiet_run[0])))
            b.start = min(gap1, max(a.end, time(i0 + quiet_run[1])))
        else:
            boundary = gap0 + CONNECTED_SPLIT * (gap1 - gap0)
            a.end = b.start = boundary

    # first word: move the start back over sound that is already there
    first = out[0]
    i = index(first.start)
    limit = index(first.start - MAX_START_EXTENSION)
    while i > limit and i - 1 < sound.size and sound[i - 1]:
        i -= 1
    first.start = min(first.start, time(i))
    # last word: extend over its remaining sound
    last = out[-1]
    i = index(last.end)
    limit = min(sound.size, index(last.end + MAX_END_EXTENSION))
    while i < limit and sound[i]:
        i += 1
    last.end = max(last.end, time(i))
    return out


def _longest_quiet_run(sound: np.ndarray) -> tuple[int, int] | None:
    if sound.size == 0:
        return None
    quiet = np.concatenate([[False], ~sound, [False]])
    edges = np.flatnonzero(quiet[1:] != quiet[:-1])
    if edges.size == 0:
        return None
    runs = list(zip(edges[0::2], edges[1::2]))
    start, stop = max(runs, key=lambda r: r[1] - r[0])
    return int(start), int(stop)
