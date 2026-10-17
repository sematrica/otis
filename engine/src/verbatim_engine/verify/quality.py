"""Pass 5: transcript quality analysis.

Looks at every segment with evidence that is independent of the words
themselves where possible, and flags what needs a second look:

* ``low_confidence`` - unclear speech: the recogniser's own probabilities are
  low, many words are uncertain, or it had to fall back to sampling;
* ``possible_hallucination`` - text where no speech was detected, text the
  acoustic model could not find in the audio, repetition loops, or the
  recogniser's own no-speech estimate;
* ``repeated_words`` - immediate repetitions ("the the"), which are often
  genuine in verbatim speech and are only pointed out;
* ``missing_speech`` - stretches where the voice-activity detector (and the
  speaker-identification model, if it ran) heard speech but no word was
  transcribed.

Overlapping speech, uncertain speakers and failed alignment are flagged by
earlier passes. Nothing is changed or removed here: flags carry the evidence
and a severity, and :mod:`.review` turns them into review priorities.

The speech map comes from Silero VAD (bundled with faster-whisper) and, when
available, the diarization's frame-level speech activity - two detectors that
do not depend on the recogniser's output.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from ..asr.merge import normalize_token
from ..audio import SAMPLE_RATE
from ..audio.wav import WavReader
from ..errors import Cancelled
from ..transcript.model import TSegment

log = logging.getLogger(__name__)

#: Speech-map resolution.
STEP = 0.02
#: Audio processed by the VAD at a time (memory stays bounded on long files).
VAD_BLOCK_SECONDS = 600.0

#: Whisper's own threshold for falling back to sampling.
LOW_AVG_LOGPROB = -1.0
#: A word is uncertain below this recogniser probability.
LOW_WORD_PROBABILITY = 0.5
#: ... and a segment is unclear when this share of its words are.
UNCERTAIN_WORD_SHARE = 0.3
UNCERTAIN_WORD_MIN = 2
#: Below this share of speech under its words, a segment is "text without speech".
NO_SPEECH_COVERAGE = 0.2
#: Whisper's own no-speech threshold.
HIGH_NO_SPEECH_PROB = 0.6
#: Whisper's own threshold for repetitive output.
HIGH_COMPRESSION_RATIO = 2.4
#: Speech without words, at least this long, is reported as missing.
MISSING_SPEECH_SECONDS = 1.0
#: Words cover the audio this far around their timestamps (timing error).
WORD_COVER_PAD = 0.25

#: Short phrases Whisper is known to produce on silence, music or noise
#: (subtitle credits, video sign-offs). On their own they are a weak signal:
#: people do say them.
KNOWN_PHRASES = (
    "thank you for watching", "thanks for watching", "thank you for listening", "please subscribe",
    "subscribe to the channel", "like and subscribe", "see you in the next video", "subtitles by",
    "subtitled by", "captions by", "transcribed by", "translated by", "amara org", "copyright",
    "all rights reserved",
)


@dataclass
class SpeechMap:
    """Where speech was detected, on a :data:`STEP` grid."""

    duration: float
    vad: np.ndarray  # bool, Silero VAD
    diarization: np.ndarray | None = None  # bool, any speaker active

    def _slice(self, start: float, end: float) -> slice:
        a = int(np.floor(max(0.0, start) / STEP))
        b = int(np.ceil(min(self.duration, end) / STEP))
        return slice(a, max(a, min(self.vad.size, b)))

    def speech(self) -> np.ndarray:
        """Speech according to either detector."""
        if self.diarization is None:
            return self.vad
        return self.vad | self.diarization

    def coverage(self, intervals: Sequence[tuple[float, float]]) -> float | None:
        """Share of the intervals' duration with detected speech (either detector)."""
        speech = self.speech()
        hit = total = 0
        for start, end in intervals:
            window = speech[self._slice(start, end)]
            hit += int(window.sum())
            total += window.size
        return None if total == 0 else hit / total

    def uncovered(self, covered: Sequence[tuple[float, float]], min_seconds: float) -> list[dict[str, Any]]:
        """Runs of VAD speech outside ``covered``, at least ``min_seconds`` long."""
        mask = self.vad.copy()
        for start, end in covered:
            mask[self._slice(start, end)] = False
        out = []
        for a, b in _runs(mask):
            if (b - a) * STEP < min_seconds:
                continue
            detectors = ["vad"]
            if self.diarization is not None and self.diarization[a:b].mean() >= 0.5:
                detectors.append("diarization")
            out.append({"start": round(a * STEP, 3), "end": round(min(self.duration, b * STEP), 3), "detectors": detectors})
        return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def build_speech_map(
    reader: WavReader,
    diarization=None,
    *,
    spans: Sequence[tuple[float, float]] | None = None,
    progress: Callable[[float, float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> SpeechMap:
    """Speech map of the recording, or only of ``spans`` (elsewhere: none)."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    duration = reader.duration
    cells = int(np.ceil(duration / STEP)) + 1
    vad = np.zeros(cells, dtype=bool)
    # fine-grained settings: short pauses are kept, only a little padding
    options = VadOptions(threshold=0.5, min_speech_duration_ms=150, min_silence_duration_ms=150, speech_pad_ms=30)
    blocks: list[tuple[float, float]] = []
    for a, b in sorted(spans) if spans is not None else [(0.0, duration)]:
        a, b = max(0.0, a), min(duration, b)
        position = a
        while position < b:
            blocks.append((position, min(b, position + VAD_BLOCK_SECONDS)))
            position = blocks[-1][1]
    total = sum(b - a for a, b in blocks) or 1.0
    done = 0.0
    for start, end in blocks:
        if should_cancel is not None and should_cancel():
            raise Cancelled("The job was cancelled.")
        audio = reader.read(start, end)
        for stamp in get_speech_timestamps(audio, options, sampling_rate=SAMPLE_RATE):
            a = int((start + stamp["start"] / SAMPLE_RATE) / STEP)
            b = int(np.ceil((start + stamp["end"] / SAMPLE_RATE) / STEP))
            vad[a:b] = True
        done += end - start
        if progress is not None:
            progress(done / total * duration, duration)
    diar = None
    if diarization is not None and diarization.num_speakers > 0:
        diar = np.zeros(cells, dtype=bool)
        active = diarization.binary.any(axis=1)
        for index in np.flatnonzero(active):
            t0, t1 = diarization.frame_bounds(int(index))
            diar[int(max(0.0, t0) / STEP): int(np.ceil(t1 / STEP))] = True
    return SpeechMap(duration, vad, diar)


# ----------------------------------------------------------------------------- detectors


def _word_spans(seg: TSegment) -> list[tuple[float, float]]:
    if seg.words:
        return [(w.start, w.end) for w in seg.words]
    return [(seg.start, seg.end)]


def segment_coverage(seg: TSegment, speech: SpeechMap) -> float | None:
    """Share of the segment's word time in which speech was detected."""
    return speech.coverage(_word_spans(seg))


def repetitions(tokens: Sequence[str]) -> list[dict[str, Any]]:
    """Immediately repeated n-grams (n = 1..4): ``[{"text", "count", "loop"}]``."""
    found: list[dict[str, Any]] = []
    i = 0
    while i < len(tokens):
        best = None
        for n in range(1, 5):
            gram = list(tokens[i : i + n])
            if len(gram) < n or not all(gram):
                break
            count = 1
            while list(tokens[i + count * n : i + (count + 1) * n]) == gram:
                count += 1
            if count >= 2 and (best is None or count * n > best[1] * best[0]):
                best = (n, count)
        if best is None:
            i += 1
            continue
        n, count = best
        loop = (n == 1 and count >= 4) or (n >= 2 and count >= 3)
        found.append({"text": " ".join(tokens[i : i + n]), "count": count, "loop": loop})
        i += n * count
    return found


def _known_phrase(text: str) -> str | None:
    normal = " ".join(t for t in (normalize_token(w) for w in re.split(r"\s+", text)) if t)
    normal = re.sub(r"[^\w ]+", " ", normal)
    for phrase in KNOWN_PHRASES:
        if phrase in normal:
            return phrase
    return None


def analyse(segments: list[TSegment], speech: SpeechMap | None, *, missing: bool = True) -> dict[str, Any]:
    """Add pass-5 flags to the segments (in place); returns counts per flag type.
    ``missing=False`` skips untranscribed-speech detection (when only some
    segments of a transcript are analysed)."""
    counts: dict[str, int] = {}

    def add(seg: TSegment, flag: dict[str, Any]) -> None:
        seg.flags.append(flag)
        counts[flag["type"]] = counts.get(flag["type"], 0) + 1

    for seg in segments:
        tokens = [normalize_token(w.text) for w in seg.words] if seg.words else [
            normalize_token(t) for t in seg.text.split()]
        probabilities = [w.probability for w in seg.words if w.probability is not None]

        # unclear speech
        reasons = []
        if seg.avg_logprob is not None and seg.avg_logprob < LOW_AVG_LOGPROB:
            reasons.append("low_average_probability")
        uncertain = sum(1 for p in probabilities if p < LOW_WORD_PROBABILITY)
        if probabilities and uncertain >= UNCERTAIN_WORD_MIN and uncertain / len(probabilities) >= UNCERTAIN_WORD_SHARE:
            reasons.append("many_uncertain_words")
        if seg.temperature is not None and seg.temperature > 0:
            reasons.append("decoder_fallback")
        if reasons:
            severe = (seg.avg_logprob is not None and seg.avg_logprob < LOW_AVG_LOGPROB - 0.3) or (
                probabilities and uncertain / len(probabilities) >= 0.5)
            add(seg, {"type": "low_confidence", "severity": "high" if severe else "medium", "reasons": reasons,
                      "avg_logprob": None if seg.avg_logprob is None else round(seg.avg_logprob, 3),
                      "uncertain_words": uncertain, "words": len(probabilities)})

        # possible hallucination
        reasons, evidence = [], {}
        coverage = speech.coverage(_word_spans(seg)) if speech is not None else None
        if coverage is not None:
            evidence["speech_coverage"] = round(coverage, 3)
            if coverage < NO_SPEECH_COVERAGE and seg.end - seg.start >= 0.3:
                reasons.append("no_speech_detected")
        failed = next((f for f in seg.flags if f.get("type") == "alignment_failed"), None)
        if failed is not None and failed.get("score") is not None and "does not match" in (failed.get("reason") or ""):
            reasons.append("text_not_in_audio")
        loops = [r for r in repetitions(tokens) if r["loop"]]
        if loops or (seg.compression_ratio is not None and seg.compression_ratio > HIGH_COMPRESSION_RATIO):
            reasons.append("repetition_loop")
        if seg.no_speech_prob is not None and seg.no_speech_prob > HIGH_NO_SPEECH_PROB:
            reasons.append("high_no_speech_probability")
        phrase = _known_phrase(seg.text)
        if phrase is not None:
            reasons.append("known_phrase")
            evidence["phrase"] = phrase
        if reasons:
            strong = {"no_speech_detected", "text_not_in_audio", "repetition_loop"}
            severity = "high" if strong & set(reasons) else ("low" if reasons == ["known_phrase"] else "medium")
            add(seg, {"type": "possible_hallucination", "severity": severity, "reasons": reasons, **evidence})

        # immediate repetitions (often genuine)
        plain = [r for r in repetitions(tokens) if not r["loop"]]
        if plain:
            add(seg, {"type": "repeated_words", "severity": "low", "repeats": plain})

    if missing and speech is not None:
        found = flag_missing_speech(segments, speech)
        if found:
            counts["missing_speech"] = len(found)
    return counts


def missing_speech(segments: Sequence[TSegment], speech: SpeechMap) -> list[dict[str, Any]]:
    """Detected speech that no word covers: stretches of at least
    :data:`MISSING_SPEECH_SECONDS`, with their detectors and a severity."""
    covered = [(a - WORD_COVER_PAD, b + WORD_COVER_PAD) for seg in segments for a, b in _word_spans(seg)]
    gaps = speech.uncovered(covered, MISSING_SPEECH_SECONDS)
    for gap in gaps:
        gap["severity"] = "high" if len(gap["detectors"]) == 2 or gap["end"] - gap["start"] >= 3.0 else "medium"
    return gaps


def flag_missing_speech(segments: Sequence[TSegment], speech: SpeechMap) -> list[dict[str, Any]]:
    """Flag detected speech that no word covers, on the segment before it
    (segments sorted by start). Returns the stretches found; without any
    segment they are only returned (the transcript is empty, not silent)."""
    gaps = missing_speech(segments, speech)
    if segments:
        starts = [seg.start for seg in segments]
        for gap in gaps:
            owner = segments[max(0, int(np.searchsorted(starts, gap["start"])) - 1)]
            owner.flags.append({"type": "missing_speech", **gap})
    return gaps
