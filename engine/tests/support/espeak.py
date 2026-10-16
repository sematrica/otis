"""Speech with known word timings, from libespeak-ng (via ctypes).

The synthesizer reports, for every word it speaks, the exact audio position
where the word starts. Ends are where the word's audio falls silent before
the next word starts. This gives connected (not word-by-word) synthetic
speech with ground-truth word boundaries for measuring timestamp accuracy.
It is synthetic speech: results say how precise the aligner is, not how it
does on real recordings.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import re
from dataclasses import dataclass

import numpy as np

AUDIO_OUTPUT_SYNCHRONOUS = 2
EVENT_WORD = 1
EVENT_END = 5
EVENT_MSG_TERMINATED = 6
ESPEAK_RATE = 1
POS_CHARACTER = 1
ESPEAKCHARS_UTF8 = 1


class _Event(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("unique_identifier", ctypes.c_uint),
        ("text_position", ctypes.c_int),
        ("length", ctypes.c_int),
        ("audio_position", ctypes.c_int),
        ("sample", ctypes.c_int),
        ("user_data", ctypes.c_void_p),
        ("id", ctypes.c_char * 8),
    ]


_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.POINTER(_Event))


@dataclass
class TimedWord:
    text: str
    start: float
    end: float


def _library():
    names = ["espeak-ng", "libespeak-ng.so.1", "libespeak-ng.1.dylib", "/opt/homebrew/lib/libespeak-ng.dylib",
             "/usr/local/lib/libespeak-ng.dylib"]
    for name in names:
        path = ctypes.util.find_library(name) if "/" not in name and not name.startswith("lib") else name
        if not path:
            continue
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    return None


_LIB = None
_RATE = None


def available() -> bool:
    global _LIB
    if _LIB is None:
        _LIB = _library() or False
    return bool(_LIB)


def synthesize_timed(text: str, voice: str = "en-us", words_per_minute: int = 160) -> tuple[np.ndarray, int, list[TimedWord]]:
    """(samples float32, sample_rate, words with true start/end seconds)."""
    global _RATE
    if not available():
        raise RuntimeError("libespeak-ng is not available")
    lib = _LIB
    if _RATE is None:
        _RATE = lib.espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 200, None, 0)
        if _RATE <= 0:
            raise RuntimeError("espeak_Initialize failed")
    chunks: list[np.ndarray] = []
    events: list[tuple[int, int, int, int]] = []  # (type, text_position, length, sample_offset)
    produced = [0]

    def on_audio(wav, count, event_ptr):
        i = 0
        while True:
            ev = event_ptr[i]
            if ev.type == 0:  # list terminated
                break
            events.append((ev.type, ev.text_position, ev.length, ev.audio_position))
            i += 1
        if count > 0 and wav:
            chunks.append(np.ctypeslib.as_array(wav, shape=(count,)).astype(np.float32) / 32768.0)
            produced[0] += count
        return 0

    callback = _CALLBACK(on_audio)
    lib.espeak_SetSynthCallback(callback)
    if lib.espeak_SetVoiceByName(voice.encode()) != 0:
        raise RuntimeError(f"espeak voice {voice!r} not available")
    lib.espeak_SetParameter(ESPEAK_RATE, words_per_minute, 0)
    data = text.encode("utf-8")
    buffer = ctypes.create_string_buffer(data)
    lib.espeak_Synth(buffer, len(data) + 1, 0, POS_CHARACTER, 0, ESPEAKCHARS_UTF8, None, None)
    lib.espeak_Synchronize()
    samples = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

    # text_position is 1-based in characters; map word events to the words of the text
    words = [(m.start(), m.group()) for m in re.finditer(r"\S+", text)]
    starts: list[tuple[str, float]] = []
    for etype, pos, length, audio_ms in events:
        if etype != EVENT_WORD:
            continue
        char = pos - 1
        match = next((w for w in words if w[0] <= char < w[0] + len(w[1])), None)
        if match is None:
            continue
        starts.append((match[1], audio_ms / 1000.0))
    total = samples.size / _RATE
    timed: list[TimedWord] = []
    for k, (word, start) in enumerate(starts):
        limit = starts[k + 1][1] if k + 1 < len(starts) else total
        timed.append(TimedWord(word, start, _sound_end(samples, _RATE, start, limit)))
    return samples, _RATE, timed


def _sound_end(samples: np.ndarray, rate: int, start: float, limit: float, threshold_db: float = -45.0) -> float:
    """Last 5 ms frame above the threshold (relative to full scale) before ``limit``."""
    a, b = int(start * rate), int(limit * rate)
    seg = samples[a:b]
    hop = max(1, rate // 200)
    if seg.size < hop:
        return limit
    frames = seg[: seg.size // hop * hop].reshape(-1, hop)
    level = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
    loud = np.nonzero(level > threshold_db)[0]
    if loud.size == 0:
        return limit
    return (a + (loud[-1] + 1) * hop) / rate
