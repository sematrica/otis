"""End-to-end tests with real Whisper weights (no mocks).

Run with e.g.::

    VERBATIM_TEST_MODEL_DIR=/path/to/faster-whisper-small uv run pytest -m model

Measured error rates are written to ``$VERBATIM_TEST_REPORT`` (JSON) when set.
Thresholds are deliberately loose regression guards; the report holds the
real numbers. Synthetic speech (espeak-ng / macOS `say`) is not representative
of real-world accuracy.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from verbatim_engine.asr.base import DecodeOptions, LoadOptions
from verbatim_engine.asr.faster_whisper_backend import FasterWhisperBackend
from verbatim_engine.metrics.wer import normalize, word_error_rate
from verbatim_engine.paths import EnginePaths
from verbatim_engine.rpc.service import EngineService
from verbatim_engine.storage import repo

from .conftest import CORPUS, SR, encode_media, read_wav, silence, synthesize, write_wav

pytestmark = pytest.mark.model

TERMINAL = ("completed", "failed", "cancelled", "interrupted")
REPORT: dict[str, dict] = {}


@pytest.fixture(scope="module", autouse=True)
def _write_report():
    yield
    target = os.environ.get("VERBATIM_TEST_REPORT")
    if target and REPORT:
        existing = json.loads(Path(target).read_text()) if Path(target).exists() else {}
        existing.update(REPORT)
        Path(target).write_text(json.dumps(existing, indent=2))


def record(name: str, reference: str, hypothesis: str, **extra) -> float:
    wer = word_error_rate(reference, hypothesis)
    REPORT[name] = {"wer": round(wer.rate, 4), "errors": wer.errors, "reference_words": wer.reference_length,
                    "substitutions": wer.substitutions, "deletions": wer.deletions, "insertions": wer.insertions,
                    "hypothesis": hypothesis, "model": os.environ.get("VERBATIM_TEST_MODEL_DIR"), **extra}
    print(f"{name}: WER {wer.rate:.3f} ({wer.errors}/{wer.reference_length})")
    return wer.rate


@pytest.fixture(scope="module")
def backend(fw_model_dir):
    b = FasterWhisperBackend()
    b.load(fw_model_dir, LoadOptions(compute_type="int8"))
    yield b
    b.close()


@pytest.fixture(scope="module")
def texts():
    return json.loads((CORPUS / "tts_texts.json").read_text())


@pytest.fixture()
def service(data_dir):
    svc = EngineService(EnginePaths.create(data_dir), lambda *_: None)
    svc.start()
    yield svc
    svc.shutdown()


def wait_job(service: EngineService, job_id: str, until=TERMINAL, timeout: float = 900.0, predicate=None) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = repo.get_job(service.db, job_id)
        if job["status"] in until or (predicate is not None and predicate(job)):
            return job
        time.sleep(0.2)
    raise TimeoutError(f"job {job_id} did not reach {until}: {repo.get_job(service.db, job_id)['status']}")


def options_for(model_dir: Path, **extra) -> dict:
    return {"backend": "faster-whisper", "model_path": str(model_dir), "mode": "balanced", **extra}


def transcript_text(service: EngineService, project_id: str) -> str:
    return " ".join(s["text"] for s in repo.get_transcript(service.db, project_id)["segments"])


def build_long_recording(tmp_path: Path, sentences: list[str], pause: float = 1.2) -> tuple[Path, str]:
    parts = []
    for i, sentence in enumerate(sentences):
        parts.append(synthesize(sentence, tmp_path / "tts", f"s{i}"))
        parts.append(silence(pause))
    path = write_wav(tmp_path / "long.wav", np.concatenate(parts))
    return path, " ".join(sentences)


def assert_each_sentence_once(text: str, sentences: list[str]) -> None:
    """Every sentence's opening words appear exactly once: catches words lost
    or duplicated at chunk boundaries (sentences start right after pauses)."""
    normalized = normalize(text)
    for sentence in sentences:
        key = " ".join(normalize(sentence).split()[:4])
        assert normalized.count(key) == 1, key


# ----------------------------------------------------------------------------- backend


def test_backend_transcribes_real_speech(backend, jfk_path, corpus):
    audio = read_wav(jfk_path)
    fractions = []
    result, effective = backend.transcribe(audio, DecodeOptions(), fractions.append)
    text = " ".join(s.text.strip() for s in result.segments)
    reference = corpus["samples"][0]["reference"]
    assert record("jfk_backend", reference, text) <= 0.10
    assert result.language == "en" and result.language_source == "detected"
    assert result.language_probability > 0.5
    words = [w for s in result.segments for w in s.words]
    assert len(words) >= 20
    assert all(0.0 <= w.start <= w.end <= 11.0 + 1e-3 for w in words)
    assert all(a.start <= b.start for a, b in zip(words, words[1:]))
    assert fractions and fractions[-1] == 1.0 and fractions == sorted(fractions)
    assert effective["beam_size"] == 5 and effective["vad"].startswith("silero")


def test_silence_produces_no_text(backend):
    rng = np.random.default_rng(1)
    audio = rng.normal(0, 0.001, 20 * SR).astype(np.float32)
    result, _ = backend.transcribe(audio, DecodeOptions())
    assert result.segments == []
    assert result.language is None  # nothing was detected, so nothing is claimed


def test_forced_language_is_reported_as_user_choice(backend, jfk_path):
    result, _ = backend.transcribe(read_wav(jfk_path), DecodeOptions(language="en", beam_size=1, best_of=1))
    assert result.language == "en" and result.language_source == "user" and result.language_probability is None


@pytest.mark.parametrize("key, lang", [("german", "de"), ("spanish", "es")])
def test_language_detection(backend, texts, tmp_path, key, lang):
    audio = synthesize(texts[key], tmp_path, key, lang=lang)
    result, _ = backend.transcribe(audio, DecodeOptions())
    text = " ".join(s.text.strip() for s in result.segments)
    record(f"tts_{key}_backend", texts[key], text, detected_language=result.language)
    assert result.language == lang


# ----------------------------------------------------------------------------- full pipeline


def test_video_file_end_to_end(service, fw_model_dir, jfk_path, corpus, tmp_path):
    video = encode_media(tmp_path / "jfk.mp4", read_wav(jfk_path), video=True)
    project = service.projects_create({"path": str(video), "title": "JFK video"})
    assert project["has_video"]
    job = service.jobs_start({"project_id": project["id"], "options": options_for(fw_model_dir)})
    job = wait_job(service, job["id"])
    assert job["status"] == "completed", job["error"]
    assert all(s["status"] == "completed" for s in job["stages"])
    assert job["effective"]["backend"] == "faster-whisper"
    assert job["effective"]["decode"]["beam_size"] == 5
    assert job["effective"]["chunking"]["chunks"] == 1
    text = transcript_text(service, project["id"])
    assert record("jfk_video_pipeline", corpus["samples"][0]["reference"], text) <= 0.10
    transcript = repo.get_transcript(service.db, project["id"], include_words=True)
    first = transcript["segments"][0]
    assert first["start"] < 1.0 and first["words"] and first["original_text"] == first["text"]
    exported = service.export_render({"project_id": project["id"], "format": "txt"})
    assert exported["suggested_filename"] == "JFK video.txt"
    assert "[00:00:00]" in exported["content"] and "fellow Americans" in exported["content"]
    out = tmp_path / "out.txt"
    service.export_write({"project_id": project["id"], "format": "txt", "path": str(out)})
    assert out.read_text(encoding="utf-8") == exported["content"]
    assert repo.get_project(service.db, project["id"])["status"] == "ready"


def test_long_recording_chunked_at_pauses(service, fw_model_dir, texts, tmp_path):
    path, reference = build_long_recording(tmp_path, texts["english_paragraphs"])
    duration = len(read_wav(path)) / SR
    project = service.projects_create({"path": str(path)})
    job = service.jobs_start({"project_id": project["id"], "options": options_for(fw_model_dir, chunk_seconds=20)})
    job = wait_job(service, job["id"])
    assert job["status"] == "completed", job["error"]
    chunks = repo.load_chunks(service.db, job["id"])
    assert len(chunks) >= 3, f"{duration:.1f}s recording"
    assert not any(c.cut_in_speech for c, _ in chunks)
    text = transcript_text(service, project["id"])
    wer = record("tts_long_chunked", reference, text, chunks=len(chunks), duration=round(duration, 1))
    assert wer <= 0.20
    assert_each_sentence_once(text, texts["english_paragraphs"])
    segments = repo.get_transcript(service.db, project["id"])["segments"]
    assert all(a["start"] <= b["start"] for a, b in zip(segments, segments[1:]))
    assert segments[-1]["end"] <= duration + 0.5


def test_abrupt_onset_keeps_first_word(service, fw_model_dir, texts, tmp_path):
    # Recordings often start mid-speech; the first word must not be dropped.
    sentences = texts["english_paragraphs"][:3]
    parts = [synthesize(s, tmp_path / "tts", f"onset{i}") for i, s in enumerate(sentences)]
    audio = np.concatenate([parts[0], silence(0.8), parts[1], silence(0.8), parts[2]])
    first = np.flatnonzero(np.abs(audio) > 0.01)[0]
    path = write_wav(tmp_path / "abrupt.wav", audio[first:])  # speech starts at sample 0
    project = service.projects_create({"path": str(path)})
    job = wait_job(service, service.jobs_start({"project_id": project["id"], "options": options_for(fw_model_dir)})["id"])
    assert job["status"] == "completed", job["error"]
    text = transcript_text(service, project["id"])
    record("tts_abrupt_onset", " ".join(sentences), text)
    assert normalize(text).startswith(" ".join(normalize(sentences[0]).split()[:3])), text
    segments = repo.get_transcript(service.db, project["id"])["segments"]
    assert segments[0]["start"] >= 0.0 and segments[0]["start"] < 0.5


def test_cuts_inside_continuous_speech(service, fw_model_dir, texts, tmp_path):
    speech = synthesize(texts["continuous"], tmp_path, "continuous")
    path = write_wav(tmp_path / "continuous.wav", speech)
    single = service.projects_create({"path": str(path), "title": "single"})
    chunked = service.projects_create({"path": str(path), "title": "chunked"})
    j1 = service.jobs_start({"project_id": single["id"], "options": options_for(fw_model_dir, chunk_seconds=3600)})
    wait_job(service, j1["id"])
    j2 = service.jobs_start({"project_id": chunked["id"], "options": options_for(fw_model_dir, chunk_seconds=20)})
    job = wait_job(service, j2["id"])
    assert job["status"] == "completed", job["error"]
    chunks = repo.load_chunks(service.db, j2["id"])
    assert len(chunks) >= 2
    single_text, chunked_text = transcript_text(service, single["id"]), transcript_text(service, chunked["id"])
    wer_single = record("tts_continuous_single", texts["continuous"], single_text)
    wer_chunked = record("tts_continuous_chunked", texts["continuous"], chunked_text, chunks=len(chunks))
    # Chunking must not add more than a couple of errors versus one pass.
    assert word_error_rate(texts["continuous"], chunked_text).errors <= word_error_rate(texts["continuous"], single_text).errors + 3
    assert wer_chunked <= 0.25 and wer_single <= 0.25
    flags = [f for s in repo.get_transcript(service.db, chunked["id"])["segments"] for f in s["flags"]]
    assert any(f["type"] == "chunk_boundary" for f in flags)


def test_cancel_then_resume_reuses_completed_chunks(service, fw_model_dir, texts, tmp_path):
    path, reference = build_long_recording(tmp_path, texts["english_paragraphs"], pause=0.8)
    project = service.projects_create({"path": str(path)})
    job = service.jobs_start({"project_id": project["id"], "options": options_for(fw_model_dir, chunk_seconds=20)})

    def one_chunk_done(_job):
        return service.db.query_one("SELECT COUNT(*) FROM chunks WHERE job_id = ? AND status = 'completed'", (job["id"],))[0] >= 1

    wait_job(service, job["id"], predicate=one_chunk_done)
    service.jobs.cancel(job["id"])
    cancelled = wait_job(service, job["id"])
    assert cancelled["status"] == "cancelled"
    done_before = {
        r["idx"]: r["completed_at"]
        for r in service.db.query("SELECT idx, completed_at FROM chunks WHERE job_id = ? AND status = 'completed'", (job["id"],))
    }
    total = service.db.query_one("SELECT COUNT(*) FROM chunks WHERE job_id = ?", (job["id"],))[0]
    assert 1 <= len(done_before) < total
    assert repo.get_project(service.db, project["id"])["status"] == "cancelled"

    service.jobs.resume(job["id"])
    resumed = wait_job(service, job["id"])
    assert resumed["status"] == "completed", resumed["error"]
    assert resumed["attempt"] == 2
    after = {r["idx"]: r["completed_at"] for r in service.db.query("SELECT idx, completed_at FROM chunks WHERE job_id = ?", (job["id"],))}
    for idx, stamp in done_before.items():
        assert after[idx] == stamp, f"chunk {idx} was re-transcribed"
    extract = next(s for s in resumed["stages"] if s["stage"] == "extract")
    assert extract["detail"] and "Reused" in extract["detail"]
    text = transcript_text(service, project["id"])
    record("tts_cancel_resume", reference, text, chunks=total)
    assert_each_sentence_once(text, texts["english_paragraphs"])


def test_interrupted_job_recovers_after_restart(fw_model_dir, texts, data_dir, tmp_path):
    path, reference = build_long_recording(tmp_path, texts["english_paragraphs"][:6], pause=0.8)
    first = EngineService(EnginePaths.create(data_dir), lambda *_: None)
    first.start()
    project = first.projects_create({"path": str(path)})
    job = first.jobs_start({"project_id": project["id"], "options": options_for(fw_model_dir, chunk_seconds=20)})
    wait_job(first, job["id"], predicate=lambda j: j["current_stage"] == "transcribe")
    first.shutdown()  # the app quits mid-job
    second = EngineService(EnginePaths.create(data_dir), lambda *_: None)
    second.start()
    try:
        assert repo.get_job(second.db, job["id"])["status"] == "interrupted"
        second.jobs.resume(job["id"])
        final = wait_job(second, job["id"])
        assert final["status"] == "completed", final["error"]
        wer = record("tts_interrupted_resume", reference, transcript_text(second, project["id"]))
        assert wer <= 0.20
    finally:
        second.shutdown()


def test_source_change_is_detected(service, fw_model_dir, jfk_path, tmp_path):
    copy = tmp_path / "copy.wav"
    copy.write_bytes(jfk_path.read_bytes())
    project = service.projects_create({"path": str(copy)})
    with open(copy, "ab") as handle:
        handle.write(b"\0" * 64)
    job = service.jobs_start({"project_id": project["id"], "options": options_for(fw_model_dir)})
    job = wait_job(service, job["id"])
    assert job["status"] == "failed" and job["error"]["code"] == "source_changed"


# ----------------------------------------------------------------------------- MLX (Apple Silicon only)


@pytest.fixture(scope="module")
def mlx_backend(mlx_model_dir):
    from verbatim_engine.asr.mlx_whisper_backend import MlxWhisperBackend
    from verbatim_engine.errors import BackendUnavailable

    available, reason = MlxWhisperBackend.availability()
    if not available:
        pytest.skip(reason)
    backend = MlxWhisperBackend()
    try:
        backend.load(mlx_model_dir, LoadOptions())
    except BackendUnavailable as exc:
        pytest.skip(exc.message)
    yield backend
    backend.close()


def test_mlx_backend_transcribes_real_speech(mlx_backend, jfk_path, corpus):
    fractions = []
    result, effective = mlx_backend.transcribe(read_wav(jfk_path), DecodeOptions(beam_size=1), fractions.append)
    text = " ".join(s.text.strip() for s in result.segments)
    assert record("jfk_mlx_backend", corpus["samples"][0]["reference"], text, backend="mlx",
                  model=os.environ.get("VERBATIM_TEST_MLX_MODEL_DIR")) <= 0.10
    assert result.language == "en" and result.language_probability > 0.5
    assert effective["beam_size"] == 1 and "greedy" in effective["beam_search"]
    words = [w for s in result.segments for w in s.words]
    assert words and all(0.0 <= w.start <= w.end <= 11.0 + 1e-3 for w in words)
    assert fractions and fractions[-1] == 1.0


def test_mlx_silence_produces_no_text(mlx_backend):
    rng = np.random.default_rng(2)
    result, _ = mlx_backend.transcribe(rng.normal(0, 0.001, 20 * SR).astype(np.float32), DecodeOptions(beam_size=1))
    assert result.segments == []


def test_mlx_pipeline_end_to_end(service, mlx_model_dir, jfk_path, corpus):
    from verbatim_engine.asr.mlx_whisper_backend import MlxWhisperBackend

    available, reason = MlxWhisperBackend.availability()
    if not available:
        pytest.skip(reason)
    project = service.projects_create({"path": str(jfk_path)})
    job = service.jobs_start({"project_id": project["id"],
                              "options": {"backend": "mlx", "model_path": str(mlx_model_dir), "mode": "maximum"}})
    job = wait_job(service, job["id"])
    if job["status"] == "failed" and job["error"]["code"] == "backend_unavailable":
        pytest.skip(job["error"]["message"])
    assert job["status"] == "completed", job["error"]
    assert job["effective"]["decode"]["beam_size"] == 1
    assert any("Beam search is not supported" in w for w in job["effective"]["warnings"])
    record("jfk_mlx_pipeline", corpus["samples"][0]["reference"], transcript_text(service, project["id"]), backend="mlx")
