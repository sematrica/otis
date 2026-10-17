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


# ----------------------------------------------------------------------------- full pipeline


def _wait(service, job_id: str, timeout: float = 1500) -> dict:
    import time

    from verbatim_engine.storage import repo

    deadline = time.monotonic() + timeout
    while repo.get_job(service.db, job_id)["status"] not in ("completed", "failed", "cancelled", "interrupted"):
        assert time.monotonic() < deadline, "job did not finish in time"
        time.sleep(0.2)
    return repo.get_job(service.db, job_id)


def test_maximum_accuracy_pipeline_and_on_demand_recheck(fw_model_dir, diarization_dir, alignment_dir,
                                                         speaker_corpus, data_dir):
    from verbatim_engine.paths import EnginePaths
    from verbatim_engine.rpc.service import EngineService
    from verbatim_engine.storage import repo
    from verbatim_engine.verify.review import LEVELS

    from tests.test_stage2_models import _install

    service = EngineService(EnginePaths.create(data_dir), lambda *_: None)
    _install(service.models, "diarization/pyannote-wespeaker", diarization_dir)
    _install(service.models, "alignment/en-conformer-small", alignment_dir)
    service.start()
    call = service.methods()
    try:
        project = service.projects_create({"path": str(speaker_corpus / "sample.wav"), "title": "Call"})
        pid = project["id"]
        options = {"backend": "faster-whisper", "model_path": str(fw_model_dir), "mode": "maximum",
                   "diarize": True, "align": True, "language": "en"}
        job = _wait(service, service.jobs_start({"project_id": pid, "options": options})["id"])
        assert job["status"] == "completed", job["error"]
        stages = {s["stage"]: s for s in job["stages"]}
        assert [s["stage"] for s in job["stages"]][-4:] == ["quality", "retranscribe", "validate", "finalize"]
        assert all(stages[k]["status"] == "completed" for k in ("quality", "retranscribe", "validate"))
        effective = job["effective"]
        assert effective["recheck"]["enabled"] is True
        assert effective["recheck"].get("acoustic_model", {"id": None})["id"] in (None, "alignment/en-conformer-small")
        print({"quality": effective["quality"]["flags"], "recheck": effective["recheck"],
               "validation": effective["validation"]})
        transcript = repo.get_transcript(service.db, pid, include_words=True)
        segments = transcript["segments"]
        assert segments and all("review_priority" in s for s in segments)
        # "needs review" means a medium or high priority (the app's review list)
        needs = sum(s["review_priority"] in ("medium", "high") for s in segments)
        assert needs == effective["validation"]["needs_review"]
        assert all(s["review_priority"] in (None, *LEVELS) for s in segments)
        starts = [s["start"] for s in segments]
        assert starts == sorted(starts)
        for seg in segments:
            assert all(seg["start"] - 1e-6 <= w["start"] <= w["end"] <= seg["end"] + 1e-6 for w in seg["words"])

        # on-demand re-check of two passages, one of them corrected by the user
        candidates = [s for s in segments if len(s["words"]) >= 4]
        assert len(candidates) >= 2
        edited, other = candidates[0], candidates[-1]
        corrected = edited["text"].strip() + " indeed"
        call["transcript.editText"]({"project_id": pid, "segment_id": edited["id"], "text": corrected})
        before = {s["id"]: s for s in repo.get_transcript(service.db, pid)["segments"]}
        recheck = call["transcript.recheck"]({"project_id": pid, "segment_ids": [edited["id"], other["id"]]})
        assert recheck["kind"] == "recheck"
        recheck = _wait(service, recheck["id"])
        assert recheck["status"] == "completed", recheck["error"]
        assert [s["stage"] for s in recheck["stages"]] == ["prepare", "retranscribe", "finalize"]
        print("re-check:", recheck["effective"]["recheck"])
        after = {s["id"]: s for s in repo.get_transcript(service.db, pid)["segments"]}
        assert after[edited["id"]]["text"].strip() == corrected  # a user's correction is never re-worded
        for sid in (edited["id"], other["id"]):
            results = [f for f in after[sid]["flags"] if f["type"] == "rechecked"]
            assert len(results) == 1, after[sid]["flags"]
            print(after[sid]["text"], "->", results[0])
        history = repo.get_transcript(service.db, pid)["history"]
        assert history["undo_summary"].startswith("Re-check 2 passages")
        undone = call["transcript.undo"]({"project_id": pid})["transcript"]
        restored = {s["id"]: s for s in undone["segments"]}
        for sid in (edited["id"], other["id"]):
            assert restored[sid]["text"] == before[sid]["text"] and restored[sid]["flags"] == before[sid]["flags"]
        call["transcript.redo"]({"project_id": pid})

        # reviewing and removing passages are undoable corrections
        reviewed = call["transcript.markReviewed"]({"project_id": pid, "segment_ids": [other["id"]]})["transcript"]
        flags = next(s for s in reviewed["segments"] if s["id"] == other["id"])["flags"]
        assert any(f["type"] == "reviewed" for f in flags)
        assert next(s for s in reviewed["segments"] if s["id"] == other["id"])["review_priority"] is None
        count = len(reviewed["segments"])
        removed = call["transcript.deleteSegment"]({"project_id": pid, "segment_id": other["id"]})["transcript"]
        assert len(removed["segments"]) == count - 1
        assert len(call["transcript.undo"]({"project_id": pid})["transcript"]["segments"]) == count

        # choosing a recorded wording, where re-checking left one
        current = repo.get_transcript(service.db, pid)["segments"]
        offered = [(s, c) for s in current if not s["text_edited"] for f in s["flags"]
                   if f["type"] in ("alternatives", "retranscribed") and not f.get("resolved")
                   for c in f.get("candidates", []) if c["text"] and c["text"] != s["text"].strip()]
        if offered:
            seg, cand = offered[0]
            chosen = call["transcript.chooseWording"]({"project_id": pid, "segment_id": seg["id"],
                                                       "source": cand["source"]})["transcript"]
            now = next(s for s in chosen["segments"] if s["id"] == seg["id"])
            assert " ".join(now["text"].split()) == " ".join(cand["text"].split())
            assert all(f.get("resolved") for f in now["flags"] if f["type"] in ("alternatives", "retranscribed"))
        else:
            print("no alternative wording was recorded on this recording")
    finally:
        service.shutdown()
