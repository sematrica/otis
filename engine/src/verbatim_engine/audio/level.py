"""Speech level normalisation for acoustic models.

Recordings of conversations often have one voice much quieter than another
(telephone legs, distant microphones). CTC acoustic models normalise their
features over the whole input, so a voice 6-10 dB below the dominant one can
end up treated as background: measured on a two-party telephone recording,
the alignment model recognised none of the quieter speaker's words until the
level was evened out. This slowly varying gain raises quiet speech towards a
common level, leaves silence and noise alone, and never attenuates.

It is applied only to audio given to analysis models; the recording itself
and what Whisper hears are unchanged.
"""

from __future__ import annotations

import numpy as np

from . import SAMPLE_RATE

HOP_SECONDS = 0.01
WINDOW_SECONDS = 0.5
TARGET_DB = -22.0
MAX_GAIN_DB = 15.0
#: Frames this far above the input's quiet level count as speech.
GATE_DB = 10.0


def even_out(audio: np.ndarray) -> np.ndarray:
    hop = int(HOP_SECONDS * SAMPLE_RATE)
    frames = audio.size // hop
    if frames < 2:
        return audio
    level = 10 * np.log10((audio[: frames * hop].astype(np.float64).reshape(frames, hop) ** 2).mean(axis=1) + 1e-12)
    floor = float(np.percentile(level, 10))
    width = max(1, int(round(WINDOW_SECONDS / HOP_SECONDS)))
    # local speech level: loudest frame within the window around each frame
    padded = np.pad(level, (width // 2, width - width // 2 - 1), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    local = windows.max(axis=1)
    gain_db = np.clip(TARGET_DB - local, 0.0, MAX_GAIN_DB)
    gain_db[local < floor + GATE_DB] = 0.0
    # smooth the gain to avoid audible steps (10 ms -> 50 ms moving average)
    kernel = np.ones(5) / 5
    gain_db = np.convolve(np.pad(gain_db, 2, mode="edge"), kernel, mode="valid")
    gain = np.repeat(10 ** (gain_db / 20), hop).astype(np.float32)
    out = audio.astype(np.float32, copy=True)
    out[: gain.size] *= gain
    if out.size > gain.size:
        out[gain.size :] *= gain[-1]
    return np.clip(out, -1.0, 1.0)
