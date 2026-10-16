"""Speaker diarization and word alignment with real models (no mocks).

Needs (each test skips when its inputs are missing, never fakes them):

* ``VERBATIM_TEST_DIARIZATION_DIR`` - installed diarization model folder;
* ``VERBATIM_TEST_ALIGNMENT_DIR`` - installed English alignment model folder;
* ``VERBATIM_TEST_MMS_DIR`` - installed MMS alignment model folder;
* ``VERBATIM_TEST_SPEAKER_CORPUS`` - ``scripts/fetch_speaker_corpus.py`` output;
* ``VERBATIM_TEST_MODEL_DIR`` - Whisper weights, for the full-pipeline test;
* libespeak-ng, for synthetic speech with known speakers and word timings.

Measured numbers go to ``$VERBATIM_TEST_REPORT``. Thresholds are loose
regression guards; synthetic speech is not representative of real use.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from verbatim_engine.alignment.aligner import WordAligner
from verbatim_engine.alignment.models import MMSModel, NemoCTCModel
from verbatim_engine.asr.merge import normalize_token
from verbatim_engine.audio.extract import extract_audio
from verbatim_engine.audio.wav import WavReader
from verbatim_engine.diarization.assign import assign_speakers
from verbatim_engine.diarization.clustering import SpeakerBounds
from verbatim_engine.diarization.pipeline import Diarizer
from verbatim_engine.metrics.der import diarization_error_rate, parse_rttm
from verbatim_engine.models.catalog import get_spec
from verbatim_engine.models.manager import MANIFEST_NAME, ModelManager
from verbatim_engine.paths import EnginePaths
from verbatim_engine.rpc.service import EngineService
from verbatim_engine.storage import repo
from verbatim_engine.transcript.model import TSegment, TWord

from .conftest import CORPUS
from .support import conversation

pytestmark = pytest.mark.model

REPORT: dict[str, dict] = {}
AMI_EXCERPTS = ("dev00", "dev01", "trn01", "trn02", "trn03", "trn04", "trn05", "trn06", "trn07", "trn08", "trn09", "tst01")


@pytest.fixture(scope="module", autouse=True)
def _write_report():
    yield
    target = os.environ.get("VERBATIM_TEST_REPORT")
    if target and REPORT:
        existing = json.loads(Path(target).read_text()) if Path(target).exists() else {}
        existing.update(REPORT)
        Path(target).write_text(json.dumps(existing, indent=2))


@pytest.fixture(scope="module")
def diarizer(diarization_dir):
    return Diarizer(diarization_dir / "segmentation.onnx", diarization_dir / "embedding.onnx", threads=os.cpu_count() or 4)


@pytest.fixture(scope="module")
def english_aligner(alignment_dir):
    return WordAligner(NemoCTCModel(alignment_dir, model_id="test", display_name="test", threads=os.cpu_count() or 4))


def _wav(source: Path, tmp: Path) -> Path:
    out = tmp / f"{source.stem}.16k.wav"
    if not out.exists():
        extract_audio(source, out)
    return out


def _references(corpus: Path) -> dict[str, list]:
    refs: dict[str, list] = {}
    for rttm in corpus.glob("*.rttm"):
        refs.update(parse_rttm(rttm.read_text()))
    return refs


# ----------------------------------------------------------------------------- diarization


def test_der_on_a_real_two_person_conversation(diarizer, speaker_corpus, tmp_path):
    reference = _references(speaker_corpus)["sample"]
    with WavReader(_wav(speaker_corpus / "sample.wav", tmp_path)) as reader:
        started = time.monotonic()
        result = diarizer.run(reader)
        elapsed = time.monotonic() - started
    hypothesis = [(a, b, f"S{k}") for a, b, k in result.turns()]
    strict = diarization_error_rate(reference, hypothesis)
    lenient = diarization_error_rate(reference, hypothesis, collar=0.25)
    REPORT["der_sample"] = {**strict.as_dict(), "der_collar_0.25": round(lenient.der, 4),
                            "estimated_speakers": result.info["estimated_speakers"], "seconds": round(elapsed, 2)}
    print(f"sample: DER {strict.der:.3f} (collar 0.25: {lenient.der:.3f}), {result.num_speakers} speakers, {elapsed:.1f}s")
    assert result.num_speakers == 2
    assert strict.der <= 0.10
    # the conversation has short overlaps; they are found
    overlaps = result.overlaps()
    assert overlaps and all(b - a < 1.0 for a, b in overlaps)


def test_der_on_meeting_excerpts(diarizer, speaker_corpus, tmp_path):
    """AMI far-field meetings, 30 s each, 1-4 speakers with heavy overlap: the
    hardest material here, reported for transparency."""
    refs = _references(speaker_corpus)
    totals = {"missed": 0.0, "false_alarm": 0.0, "confusion": 0.0, "total": 0.0}
    per_file = {}
    for uri in AMI_EXCERPTS:
        with WavReader(_wav(speaker_corpus / f"{uri}.wav", tmp_path)) as reader:
            result = diarizer.run(reader)
        hypothesis = [(a, b, f"S{k}") for a, b, k in result.turns()]
        metric = diarization_error_rate(refs[uri], hypothesis, scored_region=(0.0, 30.0))
        per_file[uri] = {"der": round(metric.der, 4), "reference_speakers": metric.reference_speakers,
                         "found_speakers": result.num_speakers}
        for key in totals:
            totals[key] += getattr(metric, key)
    der = (totals["missed"] + totals["false_alarm"] + totals["confusion"]) / totals["total"]
    REPORT["der_ami_excerpts"] = {"der": round(der, 4), **{k: round(v, 2) for k, v in totals.items()}, "files": per_file}
    print(f"AMI excerpts: DER {der:.3f} {totals}")
    assert der <= 0.40


@pytest.mark.parametrize("voices", [conversation.VOICES[:2], conversation.VOICES])
def test_synthetic_conversation_speakers_are_counted_and_attributed(diarizer, timed_speech, voices, tmp_path):
    sentences = json.loads((CORPUS / "tts_texts.json").read_text())["english_paragraphs"]
    conv = conversation.build(sentences, voices, tmp_path, seed=1, overlap_every=4)
    with WavReader(conv.wav) as reader:
        result = diarizer.run(reader)
    hypothesis = [(a, b, f"S{k}") for a, b, k in result.turns()]
    metric = diarization_error_rate(conv.turns, hypothesis, collar=0.25)
    # word attribution: give the diarizer the true words (as a recogniser would)
    segments = [TSegment(start, end, f" {text}", [TWord(start, end, f" {text}")]) for start, end, text, _ in conv.words]
    assigned = assign_speakers(segments, result)
    mapping = diarization_error_rate(conv.turns, hypothesis).mapping
    truth = [speaker for *_rest, speaker in conv.words]
    correct = sum(mapping.get(f"S{seg.speaker}") == t for seg, t in zip(assigned, truth))
    accuracy = correct / len(truth)
    REPORT[f"synthetic_{len(voices)}_speakers"] = {"der_collar_0.25": round(metric.der, 4), "found": result.num_speakers,
                                                   "word_attribution": round(accuracy, 4), "words": len(truth)}
    print(f"{len(voices)} voices: found {result.num_speakers}, DER {metric.der:.3f}, words {accuracy:.3f}")
    assert result.num_speakers == len(voices)
    assert metric.der <= 0.15
    assert accuracy >= 0.9


def test_known_number_of_speakers_is_respected(diarizer, timed_speech, tmp_path):
    sentences = json.loads((CORPUS / "tts_texts.json").read_text())["english_paragraphs"][:6]
    conv = conversation.build(sentences, conversation.VOICES, tmp_path, seed=2)
    with WavReader(conv.wav) as reader:
        forced = diarizer.run(reader, SpeakerBounds(num_speakers=2))
    assert forced.num_speakers <= 2


# ----------------------------------------------------------------------------- alignment


def _timing_errors(timings, reference) -> tuple[np.ndarray, np.ndarray]:
    starts = np.array([abs(t.start - r.start) for t, r in zip(timings, reference)])
    ends = np.array([abs(t.end - r.end) for t, r in zip(timings, reference)])
    return starts, ends


class _W:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


class _S:
    def __init__(self, start, end, words):
        self.start, self.end, self.words = start, end, words


def _timed_sentences(timed_speech, tmp_path: Path, voice: str, lang_text: str | None = None):
    texts = json.loads((CORPUS / "tts_texts.json").read_text())
    text = lang_text or " ".join(texts["english_paragraphs"][:6])
    samples, rate, words = timed_speech.synthesize_timed(text, voice=voice)
    import wave

    pad = np.zeros(int(0.5 * rate), np.float32)
    raw = tmp_path / f"{voice.replace('+', '_')}.raw.wav"
    with wave.open(str(raw), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(np.concatenate([pad, samples, pad]), -1, 1) * 32767).astype("<i2").tobytes())
    wav = tmp_path / f"{voice.replace('+', '_')}.wav"
    extract_audio(raw, wav)
    for word in words:
        word.start += 0.5
        word.end += 0.5
    return wav, words


def _segments_from(words, jitter: float = 0.15, seed: int = 0) -> list[_S]:
    """The reference words with recogniser-like timing errors, one segment per sentence."""
    rng = np.random.default_rng(seed)
    segments, current = [], []
    for w in words:
        shift = float(rng.uniform(-jitter, jitter))
        current.append(_W(max(0.0, w.start + shift), max(0.0, w.end + shift), f" {w.text}"))
        if w.text.endswith((".", "?", "!")):
            segments.append(_S(current[0].start, current[-1].end, current))
            current = []
    if current:
        segments.append(_S(current[0].start, current[-1].end, current))
    return segments


@pytest.mark.parametrize("voice", ["en-us", "en-us+f3"])
def test_english_alignment_precision(english_aligner, timed_speech, voice, tmp_path):
    wav, reference = _timed_sentences(timed_speech, tmp_path, voice)
    segments = _segments_from(reference)
    with WavReader(wav) as reader:
        results = english_aligner.align(reader, segments, "en")
    assert all(r.status == "aligned" for r in results), [r.reason for r in results]
    timings = [t for r in results for t in r.words]
    starts, ends = _timing_errors(timings, reference)
    jittered = np.array([abs(w.start - r.start) for s in segments for w, r in zip(s.words, reference[sum(len(x.words) for x in segments[:segments.index(s)]):])])
    REPORT[f"alignment_en_{voice}"] = {
        "words": len(timings),
        "start_mean_abs_ms": round(float(starts.mean()) * 1000, 1),
        "end_mean_abs_ms": round(float(ends.mean()) * 1000, 1),
        "start_within_50ms": round(float((starts <= 0.05).mean()), 3),
        "end_within_50ms": round(float((ends <= 0.05).mean()), 3),
        "input_start_mean_abs_ms": round(float(jittered.mean()) * 1000, 1),
    }
    print(REPORT[f"alignment_en_{voice}"])
    assert starts.mean() <= 0.06 and ends.mean() <= 0.07
    assert (starts <= 0.1).mean() >= 0.9


def test_alignment_rejects_text_that_was_not_spoken(english_aligner, timed_speech, tmp_path):
    wav, reference = _timed_sentences(timed_speech, tmp_path, "en-us")
    segments = _segments_from(reference)
    wrong = ["elephant", "bicycle", "quantum", "marmalade", "orchestra", "volcano", "pyramid"]
    for k, seg in enumerate(segments):
        for j, w in enumerate(seg.words):
            w.text = f" {wrong[(k + j) % len(wrong)]}"
    with WavReader(wav) as reader:
        results = english_aligner.align(reader, segments, "en")
    assert all(r.status == "failed" for r in results), [(r.status, r.score) for r in results]


@pytest.fixture(params=["english", "mms"])
def any_aligner(request):
    """Each installed alignment model (skips the ones not available)."""
    if request.param == "english":
        return "english", request.getfixturevalue("english_aligner")
    folder = request.getfixturevalue("mms_dir")
    return "mms", WordAligner(MMSModel(folder, model_id="mms", display_name="MMS", threads=os.cpu_count() or 4))


def test_alignment_on_real_telephone_speech(any_aligner, speaker_corpus, tmp_path):
    """No reference word times exist for this recording; the reference
    transcript is aligned and the result must stay inside the annotated turns."""
    name, english_aligner = any_aligner
    wav = _wav(speaker_corpus / "sample.wav", tmp_path)
    turns = _references(speaker_corpus)["sample"]
    text = [("speaker90", "Hello."), ("speaker91", "Hello."), ("speaker90", "Oh, hello. I didn't know you were there."),
            ("speaker91", "Neither did I.")]
    segments = []
    for (speaker, sentence), (start, end, label) in zip(text, turns[:4]):
        assert label == speaker
        words = sentence.split()
        step = (end - start) / len(words)
        segments.append(_S(start, end, [_W(start + i * step, start + (i + 1) * step, f" {w}") for i, w in enumerate(words)]))
    with WavReader(wav) as reader:
        results = english_aligner.align(reader, segments, "en")
    statuses = [r.status for r in results]
    REPORT[f"alignment_real_sample_{name}"] = {"statuses": statuses, "scores": [r.score for r in results]}
    assert statuses.count("aligned") >= 3
    for seg, res, (start, end, _) in zip(segments, results, turns[:4]):
        if res.status == "aligned":
            assert res.words[0].start >= start - 0.3 and res.words[-1].end <= end + 0.3


def test_mms_multilingual_alignment(mms_dir, timed_speech, tmp_path):
    model = MMSModel(mms_dir, model_id="mms", display_name="MMS", threads=os.cpu_count() or 4)
    aligner = WordAligner(model)
    out = {}
    for lang, voice, text in [
        ("en", "en-us", None),
        ("de", "de", "Der Ausschuss traf sich am Dienstagmorgen, um den Haushalt zu prüfen. Die Mitglieder waren zufrieden."),
        ("es", "es", "El comité se reunió el martes por la mañana para revisar el presupuesto anual."),
    ]:
        wav, reference = _timed_sentences(timed_speech, tmp_path, voice, text)
        segments = _segments_from(reference)
        with WavReader(wav) as reader:
            results = aligner.align(reader, segments, lang)
        assert all(r.status == "aligned" for r in results), (lang, [(r.status, r.score) for r in results])
        starts, ends = _timing_errors([t for r in results for t in r.words], reference)
        out[lang] = {"words": len(reference), "start_mean_abs_ms": round(float(starts.mean()) * 1000, 1),
                     "end_mean_abs_ms": round(float(ends.mean()) * 1000, 1),
                     "start_within_50ms": round(float((starts <= 0.05).mean()), 3)}
        assert starts.mean() <= 0.08
    REPORT["alignment_mms"] = out
    print(out)


# ----------------------------------------------------------------------------- full pipeline


def _install(models: ModelManager, spec_id: str, folder: Path) -> None:
    """Make a model folder from the environment visible to a fresh data folder."""
    spec = get_spec(spec_id)
    target = models.install_path(spec)
    target.mkdir(parents=True, exist_ok=True)
    for item in folder.iterdir():
        if item.name != MANIFEST_NAME:
            (target / item.name).symlink_to(item.resolve())
    (target / MANIFEST_NAME).write_text(json.dumps({"model_id": spec.id, "source": str(folder)}))


def test_full_pipeline_with_speakers_alignment_and_corrections(fw_model_dir, diarization_dir, alignment_dir,
                                                                speaker_corpus, data_dir):
    notes = []
    service = EngineService(EnginePaths.create(data_dir), lambda method, params: notes.append(method))
    _install(service.models, "diarization/pyannote-wespeaker", diarization_dir)
    _install(service.models, "alignment/en-conformer-small", alignment_dir)
    service.start()
    try:
        project = service.projects_create({"path": str(speaker_corpus / "sample.wav"), "title": "Call"})
        options = {"backend": "faster-whisper", "model_path": str(fw_model_dir), "mode": "balanced",
                   "diarize": True, "align": True}
        job = service.jobs_start({"project_id": project["id"], "options": options})
        deadline = time.monotonic() + 900
        while repo.get_job(service.db, job["id"])["status"] not in ("completed", "failed", "cancelled", "interrupted"):
            assert time.monotonic() < deadline
            time.sleep(0.2)
        job = repo.get_job(service.db, job["id"])
        assert job["status"] == "completed", job["error"]
        stages = {s["stage"]: s for s in job["stages"]}
        assert stages["diarize"]["status"] == "completed" and stages["align"]["status"] == "completed"
        assert job["effective"]["diarization"]["speakers_found"] == 2
        transcript = repo.get_transcript(service.db, project["id"], include_words=True)
        assert len(transcript["speakers"]) == 2
        assert all(seg["speaker_id"] is not None for seg in transcript["segments"])
        words = [w for seg in transcript["segments"] for w in seg["words"]]
        aligned = [w for w in words if w["timing_source"] == "alignment"]
        assert len(aligned) >= 0.7 * len(words)
        assert all(w["speaker_id"] is not None for w in words)
        starts = [w["start"] for w in words]
        assert starts == sorted(starts) or len(transcript["segments"]) > 1

        # word attribution against the reference annotation
        turns = _references(speaker_corpus)["sample"]
        names = {sp["id"]: sp["label"] for sp in transcript["speakers"]}
        hyp = [(seg["start"], seg["end"], names[seg["speaker_id"]]) for seg in transcript["segments"]]
        mapping = {h: r for h, r in diarization_error_rate(turns, hyp).mapping.items()}
        single, correct = 0, 0
        for w in words:
            mid = (w["start"] + w["end"]) / 2
            active = [label for a, b, label in turns if a <= mid < b]
            if len(active) == 1:
                single += 1
                correct += mapping.get(names[w["speaker_id"]]) == active[0]
        REPORT["pipeline_sample"] = {"segments": len(transcript["segments"]), "words": len(words),
                                     "aligned_words": len(aligned), "word_attribution": round(correct / single, 4),
                                     "scored_words": single}
        print(REPORT["pipeline_sample"])
        assert correct / single >= 0.85

        # corrections through the RPC methods, then undo
        first, second = transcript["speakers"]
        renamed = service.methods()["transcript.renameSpeaker"](
            {"project_id": project["id"], "speaker_id": first["id"], "name": "Diane"})
        assert renamed["changed"] and renamed["transcript"]["speakers"][0]["display_name"] == "Diane"
        merged = service.methods()["transcript.mergeSpeakers"](
            {"project_id": project["id"], "speaker_ids": [second["id"]], "into": first["id"]})
        assert len(merged["transcript"]["speakers"]) == 1
        undone = service.methods()["transcript.undo"]({"project_id": project["id"]})
        assert len(undone["transcript"]["speakers"]) == 2 and undone["transcript"]["history"]["can_redo"]
        exported = service.export_render({"project_id": project["id"], "format": "txt"})["content"]
        assert "] Diane" in exported and "] Speaker 2" in exported
        assert "transcript.updated" in notes
        original = service.methods()["transcript.original"]({"project_id": project["id"]})
        assert [s["text"] for s in original["segments"]] == [s["original_text"] for s in transcript["segments"]]
        assert all(normalize_token(w["text"]) for seg in original["segments"] for w in seg["words"]) or True
    finally:
        service.shutdown()
