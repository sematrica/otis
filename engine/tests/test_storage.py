"""SQLite schema, repository functions, options and the model manager."""

from __future__ import annotations

import json

import pytest

from verbatim_engine.asr.base import Word
from verbatim_engine.asr.merge import MergedSegment
from verbatim_engine.errors import InvalidParams, ModelNotInstalled, NotFound
from verbatim_engine.jobs.options import MODES, normalize_options, resolve
from verbatim_engine.media.probe import probe
from verbatim_engine.models.catalog import CATALOG, get_spec
from verbatim_engine.models.manager import MANIFEST_NAME, ModelManager
from verbatim_engine.paths import EnginePaths
from verbatim_engine.storage import repo
from verbatim_engine.storage.db import LATEST_VERSION, Database
from verbatim_engine.transcript.model import TSegment


@pytest.fixture()
def env(data_dir):
    paths = EnginePaths.create(data_dir)
    db = Database(paths.db_path)
    yield paths, db
    db.close()


def test_migrations_are_idempotent(env):
    paths, db = env
    assert db.schema_version == LATEST_VERSION
    again = Database(paths.db_path)
    assert again.schema_version == LATEST_VERSION
    again.close()
    assert db.query_one("PRAGMA journal_mode")[0] == "wal"
    assert db.query_one("PRAGMA foreign_keys")[0] == 1


def test_project_job_transcript_lifecycle(env, jfk_path):
    paths, db = env
    project = repo.create_project(db, probe(jfk_path), "JFK")
    assert project["status"] == "new" and project["duration"] == pytest.approx(11.0, abs=0.01)
    job = repo.create_job(db, project["id"], "balanced", normalize_options({}))
    assert job["status"] == "queued"
    assert [s["stage"] for s in job["stages"]] == [s for s, _ in repo.STAGES]
    assert repo.get_project(db, project["id"])["status"] == "queued"

    repo.mark_job_running(db, job["id"], 1234)
    repo.start_stage(db, job["id"], "transcribe", unit="seconds", total=100.0, done=0.0)
    repo.update_stage(db, job["id"], "transcribe", done=25.0)
    snap = repo.get_job(db, job["id"])
    stage = next(s for s in snap["stages"] if s["stage"] == "transcribe")
    assert stage["fraction"] == pytest.approx(0.25)
    assert stage["eta_seconds"] is None  # not enough elapsed time measured yet
    assert next(s for s in snap["stages"] if s["stage"] == "load_model")["fraction"] is None

    segs = [
        TSegment.from_merged(MergedSegment(
            0.0, 2.0, "Hello there.", [Word(0.0, 0.5, " Hello", 0.9), Word(0.5, 2.0, " there.", 0.8)],
            -0.2, 0.01, 1.2, 0.0, 0, "en", [])),
        TSegment.from_merged(MergedSegment(2.5, 4.0, "Second line.", None, -0.3, 0.02, 1.1, 0.0, 0, "en",
                                           [{"type": "chunk_boundary", "method": "aligned"}])),
    ]
    assert repo.replace_transcript(db, project["id"], job["id"], segs, "en") == 2
    repo.set_effective(db, job["id"], {"backend": "faster-whisper"})
    repo.finish_job(db, job["id"], "completed")
    transcript = repo.get_transcript(db, project["id"], include_words=True)
    assert [s["text"] for s in transcript["segments"]] == ["Hello there.", "Second line."]
    assert transcript["segments"][0]["original_text"] == "Hello there."
    assert len(transcript["segments"][0]["words"]) == 2
    assert transcript["segments"][1]["flags"][0]["method"] == "aligned"
    assert transcript["effective"] == {"backend": "faster-whisper"}
    assert repo.get_project(db, project["id"])["status"] == "ready"

    # A later failed re-run keeps the project usable.
    job2 = repo.create_job(db, project["id"], "fast", normalize_options({"mode": "fast"}))
    repo.mark_job_running(db, job2["id"], 99)
    repo.finish_job(db, job2["id"], "failed", error_code="x", error_message="boom")
    assert repo.get_project(db, project["id"])["status"] == "ready"
    assert repo.get_transcript(db, project["id"])["effective"] == {"backend": "faster-whisper"}

    # Deleting cascades.
    job_ids = repo.delete_project(db, project["id"])
    assert set(job_ids) == {job["id"], job2["id"]}
    assert db.query_one("SELECT COUNT(*) FROM segments")[0] == 0
    assert db.query_one("SELECT COUNT(*) FROM words")[0] == 0
    assert db.query_one("SELECT COUNT(*) FROM job_stages")[0] == 0
    with pytest.raises(NotFound):
        repo.get_project(db, project["id"])


def test_settings_round_trip(env):
    _, db = env
    assert repo.get_setting(db, "x", "default") == "default"
    repo.set_setting(db, "x", {"a": 1})
    repo.set_setting(db, "x", {"a": 2})
    assert repo.get_setting(db, "x") == {"a": 2}


def test_job_dir_rejects_traversal(env):
    paths, _ = env
    with pytest.raises(ValueError):
        paths.job_dir("../../etc")


# ----------------------------------------------------------------------------- options


def test_option_validation():
    assert normalize_options({})["mode"] == "balanced"
    assert normalize_options({"language": "auto"})["language"] is None
    for bad in ({"mode": "ultra"}, {"backend": "cuda"}, {"language": "xx"}, {"chunk_seconds": 5},
                {"model_path": "/m"}, {"unknown": 1}, {"model_id": "mlx/small", "backend": "faster-whisper"}):
        with pytest.raises((InvalidParams, NotFound)):
            normalize_options(bad)
    assert normalize_options({"model_id": "mlx/small"})["backend"] == "mlx"


def test_resolve_reports_missing_model(env):
    paths, db = env
    manager = ModelManager(db, paths)
    with pytest.raises(ModelNotInstalled) as info:
        resolve(normalize_options({"backend": "faster-whisper", "mode": "maximum"}), manager)
    assert info.value.data["model_id"] == "faster-whisper/large-v3"


def test_resolve_custom_model_folder(env, tmp_path):
    paths, db = env
    manager = ModelManager(db, paths)
    folder = tmp_path / "my-model"
    folder.mkdir()
    with pytest.raises(ModelNotInstalled) as info:
        resolve(normalize_options({"backend": "faster-whisper", "model_path": str(folder)}), manager)
    assert "model.bin" in info.value.message
    for name in ("model.bin", "config.json", "tokenizer.json"):
        (folder / name).write_text("{}")
    resolved = resolve(normalize_options({"backend": "faster-whisper", "model_path": str(folder), "mode": "fast"}), manager)
    assert resolved.backend == "faster-whisper"
    assert resolved.decode.beam_size == 1 and resolved.load.compute_type == "int8"
    maximum = resolve(normalize_options({"backend": "faster-whisper", "model_path": str(folder), "mode": "maximum"}), manager)
    assert maximum.decode.beam_size == 5 and maximum.load.compute_type == "float32"
    # Maximum Accuracy re-checks doubtful passages; Fast does not, unless asked to
    assert maximum.recheck and not resolved.recheck
    assert "Maximum Accuracy" in resolved.recheck_skip_reason
    forced = resolve(normalize_options({"backend": "faster-whisper", "model_path": str(folder), "mode": "fast",
                                        "recheck": True}), manager)
    assert forced.recheck
    off = resolve(normalize_options({"backend": "faster-whisper", "model_path": str(folder), "mode": "maximum",
                                     "recheck": False}), manager)
    assert not off.recheck and "turned off" in off.recheck_skip_reason
    assert maximum.describe()["recheck"]["enabled"] is True


def test_modes_cover_spec():
    assert set(MODES) == {"fast", "balanced", "maximum"}
    assert MODES["maximum"].family == "large-v3"


# ----------------------------------------------------------------------------- models


def test_catalog_is_consistent():
    ids = [s.id for s in CATALOG]
    assert len(ids) == len(set(ids))
    families = {s.family for s in CATALOG}
    assert families == {"small", "medium", "large-v3", "large-v3-turbo"}
    for spec in CATALOG:
        assert spec.approx_download_bytes > 0 and spec.memory_estimates
        assert spec.id == f"{spec.backend}/{spec.family}"


def test_model_status_requires_manifest_and_files(env):
    paths, db = env
    manager = ModelManager(db, paths)
    spec = get_spec("faster-whisper/small")
    assert manager.status(spec)["installed"] is False
    folder = manager.install_path(spec)
    folder.mkdir(parents=True)
    for name in ("model.bin", "config.json", "tokenizer.json"):
        (folder / name).write_text("x")
    assert manager.status(spec)["installed"] is False  # no manifest: not a verified download
    (folder / MANIFEST_NAME).write_text(json.dumps({"revision": "abc"}))
    status = manager.status(spec)
    assert status["installed"] and status["revision"] == "abc"
    assert manager.require_installed(spec) == folder
    manager.delete(spec)
    assert not folder.exists()


def test_models_dir_can_be_moved(env, tmp_path):
    paths, db = env
    manager = ModelManager(db, paths)
    external = tmp_path / "ExternalSSD" / "VerbatimModels"
    assert manager.set_models_dir(str(external)) == external.resolve()
    assert manager.install_path(get_spec("mlx/large-v3")) == external.resolve() / "mlx" / "large-v3"
    with pytest.raises(InvalidParams):
        manager.set_models_dir("relative/path")
    manager.set_models_dir(None)
    assert manager.models_dir() == paths.default_models_dir
