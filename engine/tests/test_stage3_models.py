"""Stage 3 tests with real models (alignment model; Whisper where noted).

Speech is synthesized with espeak-ng so the true words are known exactly.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from verbatim_engine.alignment.models import load_model
from verbatim_engine.alignment.score import PassageScorer
from verbatim_engine.verify.compare import DECISION_MARGIN
from verbatim_engine.audio.extract import extract_audio
from verbatim_engine.audio.wav import WavReader
from verbatim_engine.models.catalog import get_spec

pytestmark = pytest.mark.model


def speech_16k(timed_speech, text: str, tmp_path: Path, voice: str = "en-us", pad: float = 0.5):
    """Synthesized speech resampled to 16 kHz by the engine's own decoder,
    with ``pad`` seconds of silence on both sides, and its word timings."""
    samples, rate, words = timed_speech.synthesize_timed(text, voice=voice)
    silence = np.zeros(int(pad * rate), np.float32)
    raw = tmp_path / f"{abs(hash((text, voice)))}.raw.wav"
    with wave.open(str(raw), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(np.concatenate([silence, samples, silence]), -1, 1) * 32767).astype("<i2").tobytes())
    out = raw.with_suffix(".16k.wav")
    extract_audio(raw, out)
    with WavReader(out) as reader:
        audio = reader.read(0.0, reader.duration)
    for word in words:
        word.start += pad
        word.end += pad
    return audio, words


@pytest.fixture(scope="module")
def english_scorer(alignment_dir):
    model = load_model(get_spec("alignment/en-conformer-small"), alignment_dir, threads=2)
    return PassageScorer(model, "en")


def _score(scorer, emissions, text: str) -> float:
    result = scorer.score(emissions, text.split())
    assert result is not None, text
    return result.log_likelihood


JFK = "and so my fellow americans ask not what your country can do for you ask what you can do for your country"
#: Fillers for substitutions and insertions (common words and a few others).
FILLERS = "the a of and to in house time make good river paper over under".split()


def _variants(words: list[str], seed: int = 0) -> dict[str, list[list[str]]]:
    rng = np.random.default_rng(seed)
    out: dict[str, list[list[str]]] = {"substitution": [], "deletion": [], "insertion": [], "repetition": []}
    for i in range(len(words)):
        other = [f for f in FILLERS if f != words[i]]
        out["substitution"].append(words[:i] + [str(rng.choice(other))] + words[i + 1:])
        out["deletion"].append(words[:i] + words[i + 1:])
        out["insertion"].append(words[:i] + [str(rng.choice(FILLERS))] + words[i:])
        out["repetition"].append(words[: i + 1] + [words[i]] + words[i + 1:])
    return out


def test_acoustic_score_prefers_the_spoken_words(english_scorer, jfk_path, tmp_path):
    """Real speech: every single-word corruption of the true text is scored
    against the audio. (Synthetic espeak speech cannot be used here: the
    acoustic model does not recognise its voice, see the Stage 3 report.)"""
    wav = tmp_path / "jfk.wav"
    extract_audio(jfk_path, wav)
    with WavReader(wav) as reader:
        emissions = english_scorer.emissions(reader.read(0.0, reader.duration))
    words = JFK.split()
    truth = english_scorer.score(emissions, words).log_likelihood
    leads = {kind: [truth - english_scorer.score(emissions, v).log_likelihood for v in variants]
             for kind, variants in _variants(words).items()}
    print({k: (round(min(v), 2), round(float(np.median(v)), 2)) for k, v in leads.items()})
    for kind in ("substitution", "insertion", "repetition"):
        assert min(leads[kind]) > 0, kind
    # Dropping a short, quickly spoken word can score about as well as the
    # truth (CTC favours blanks), but never by a clear margin.
    assert min(leads["deletion"]) > -DECISION_MARGIN
    assert sum(lead > 0 for lead in leads["deletion"]) >= 0.85 * len(words)
    empty = english_scorer.score(emissions, [])
    assert truth - empty.log_likelihood > 10 * DECISION_MARGIN  # words explain speech far better than silence


def test_silence_is_best_explained_by_no_words(english_scorer):
    rng = np.random.default_rng(0)
    audio = (rng.normal(size=48_000) * 0.003).astype(np.float32)  # 3 s of faint noise
    emissions = english_scorer.emissions(audio)
    empty = english_scorer.score(emissions, []).log_likelihood
    for text in ("thank you", "thank you for watching", "you"):
        assert empty > _score(english_scorer, emissions, text), text


def test_numbers_are_scored_by_their_spoken_form(english_scorer, timed_speech, tmp_path):
    audio, _ = speech_16k(timed_speech, "it happened in nineteen eighty four and cost five dollars", tmp_path)
    emissions = english_scorer.emissions(audio)
    spelled = _score(english_scorer, emissions, "it happened in nineteen eighty four and cost five dollars")
    digits = _score(english_scorer, emissions, "it happened in 1984 and cost $5")
    cardinal = _score(english_scorer, emissions, "it happened in one thousand nine hundred eighty four and cost five dollars")
    assert digits == pytest.approx(max(spelled, cardinal), abs=1e-6)  # the best reading is used
    assert english_scorer.score(emissions, ["it", "happened", "in", "≈"]) is None  # cannot be spelled
