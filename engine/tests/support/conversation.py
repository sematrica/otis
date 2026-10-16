"""Synthetic multi-speaker conversations with exact ground truth.

Sentences are spoken by different espeak-ng voices (distinct timbres: a
male, a female and a deeper male variant), separated by pauses, optionally
with one sentence overlapping the end of the previous one. The reference
records who speaks when (``turns``) and every word's speaker and timing.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import espeak

VOICES = ("en-us", "en-us+f3", "en-us+m7")


@dataclass
class Conversation:
    wav: Path
    turns: list[tuple[float, float, str]]
    words: list[tuple[float, float, str, str]] = field(default_factory=list)  # (start, end, text, speaker)
    texts: list[tuple[str, str]] = field(default_factory=list)  # (speaker, sentence)


def build(sentences: list[str], voices: tuple[str, ...], out_dir: Path, *, seed: int = 0,
          overlap_every: int = 0, name: str = "conversation") -> Conversation:
    from verbatim_engine.audio.extract import extract_audio

    rng = np.random.default_rng(seed)
    order = [voices[i % len(voices)] for i in range(len(sentences))]
    rng.shuffle(order)
    rate = None
    track = np.zeros(0, np.float32)
    turns: list[tuple[float, float, str]] = []
    words: list[tuple[float, float, str, str]] = []
    cursor = 0.5
    for k, (sentence, voice) in enumerate(zip(sentences, order)):
        samples, rate, timed = espeak.synthesize_timed(sentence, voice=voice)
        if overlap_every and k and k % overlap_every == 0:
            cursor -= 0.6  # start before the previous speaker has finished
        start_sample = int(cursor * rate)
        needed = start_sample + samples.size
        if track.size < needed:
            track = np.concatenate([track, np.zeros(needed - track.size, np.float32)])
        track[start_sample:needed] += samples
        turns.append((cursor + timed[0].start, cursor + timed[-1].end, voice))
        words.extend((cursor + w.start, cursor + w.end, w.text, voice) for w in timed)
        cursor += samples.size / rate + float(rng.uniform(0.3, 0.9))
    track = np.concatenate([track, np.zeros(int(0.5 * rate), np.float32)])
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / f"{name}.raw.wav"
    with wave.open(str(raw), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(track, -1, 1) * 32767).astype("<i2").tobytes())
    wav = out_dir / f"{name}.wav"
    extract_audio(raw, wav)
    return Conversation(wav, turns, words, list(zip(order, sentences)))
