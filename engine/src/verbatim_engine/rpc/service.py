"""RPC method implementations. See docs/ENGINE_PROTOCOL.md."""

from __future__ import annotations

import logging
import os
import platform
import sys
from pathlib import Path
from typing import Any, Callable

from .. import PROTOCOL_VERSION, __version__
from ..asr.registry import describe_backends
from ..errors import InvalidParams, NotFound
from ..export import FORMATS, render, suggested_filename, write_atomic
from ..jobs.manager import JobManager
from ..jobs.options import modes_payload, normalize_options, resolve
from ..media.probe import probe
from ..models.catalog import get_spec
from ..models.downloads import DownloadManager
from ..models.manager import ModelManager
from ..paths import EnginePaths
from ..storage import repo
from ..storage.db import Database
from ..transcript.edit import TranscriptEditor
from ..util.hardware import describe as describe_hardware

log = logging.getLogger(__name__)

Notify = Callable[[str, dict[str, Any]], None]


def _require(params: dict[str, Any], key: str, kind: type = str) -> Any:
    value = params.get(key)
    if value is None or (kind is str and not str(value).strip()):
        raise InvalidParams(f"Missing required parameter '{key}'.")
    if not isinstance(value, kind):
        raise InvalidParams(f"Parameter '{key}' must be of type {kind.__name__}.")
    return value


def _int_list(params: dict[str, Any], key: str) -> list[int]:
    value = params.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        raise InvalidParams(f"Parameter '{key}' must be a non-empty list of integers.")
    return value


class EngineService:
    def __init__(self, paths: EnginePaths, notify: Notify, worker_command: list[str] | None = None):
        self.paths = paths
        self.notify = notify
        self.db = Database(paths.db_path)
        self.models = ModelManager(self.db, paths)
        self.jobs = JobManager(self.db, paths, notify, worker_command)
        self.downloads = DownloadManager(paths, notify, worker_command)
        self.editor = TranscriptEditor(self.db)
        self.shutdown_requested = False

    def start(self) -> None:
        self.jobs.start()

    def shutdown(self) -> None:
        self.downloads.shutdown()
        self.jobs.shutdown()
        self.db.close()

    def methods(self) -> dict[str, Callable[[dict[str, Any]], Any]]:
        return {
            "engine.info": self.engine_info,
            "engine.shutdown": self.engine_shutdown,
            "media.probe": self.media_probe,
            "projects.create": self.projects_create,
            "projects.list": self.projects_list,
            "projects.get": self.projects_get,
            "projects.rename": self.projects_rename,
            "projects.delete": self.projects_delete,
            "transcription.modes": lambda _p: {"modes": modes_payload()},
            "transcription.preflight": self.transcription_preflight,
            "jobs.start": self.jobs_start,
            "jobs.cancel": lambda p: self.jobs.cancel(_require(p, "job_id")),
            "jobs.resume": lambda p: self.jobs.resume(_require(p, "job_id")),
            "jobs.get": lambda p: repo.get_job(self.db, _require(p, "job_id")),
            "jobs.list": self.jobs_list,
            "transcript.get": self.transcript_get,
            "transcript.original": self.transcript_original,
            "transcript.history": lambda p: {"edits": self.editor.history(_require(p, "project_id"))},
            "transcript.renameSpeaker": lambda p: self._edit(p, lambda pid: self.editor.rename_speaker(
                pid, _require(p, "speaker_id", int), p.get("name"))),
            "transcript.createSpeaker": lambda p: self._edit(p, lambda pid: self.editor.create_speaker(
                pid, p.get("name"))[0]),
            "transcript.mergeSpeakers": lambda p: self._edit(p, lambda pid: self.editor.merge_speakers(
                pid, _int_list(p, "speaker_ids"), _require(p, "into", int))),
            "transcript.assignSpeaker": lambda p: self._edit(p, lambda pid: self.editor.assign_speaker(
                pid, _int_list(p, "segment_ids"), p.get("speaker_id"), p.get("new_speaker_name"))[0]),
            "transcript.splitSegment": lambda p: self._edit(p, lambda pid: self.editor.split_segment(
                pid, _require(p, "segment_id", int), _require(p, "word_index", int))[0]),
            "transcript.mergeSegments": lambda p: self._edit(p, lambda pid: self.editor.merge_segments(
                pid, _int_list(p, "segment_ids"))),
            "transcript.editText": lambda p: self._edit(p, lambda pid: self.editor.edit_text(
                pid, _require(p, "segment_id", int), _require(p, "text"))),
            "transcript.markUncertain": lambda p: self._edit(p, lambda pid: self.editor.mark_uncertain(
                pid, _require(p, "segment_id", int), bool(p.get("uncertain", True)), p.get("note"))),
            "transcript.undo": lambda p: self._edit(p, self.editor.undo),
            "transcript.redo": lambda p: self._edit(p, self.editor.redo),
            "export.formats": lambda _p: {"formats": list(FORMATS)},
            "export.render": self.export_render,
            "export.write": self.export_write,
            "models.list": self.models_list,
            "models.download": lambda p: self.downloads.start(_require(p, "model_id")),
            "models.cancelDownload": self.models_cancel_download,
            "models.delete": self.models_delete,
            "settings.get": self.settings_get,
            "settings.set": self.settings_set,
        }

    # ------------------------------------------------------------------ engine

    def engine_info(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {
            "version": __version__,
            "protocol_version": PROTOCOL_VERSION,
            "pid": os.getpid(),
            "python": {"version": platform.python_version(), "executable": sys.executable},
            "data_dir": str(self.paths.data_dir),
            "db_path": str(self.paths.db_path),
            "schema_version": self.db.schema_version,
            "hardware": describe_hardware(),
            "backends": describe_backends(),
        }

    def engine_shutdown(self, _params: dict[str, Any]) -> dict[str, Any]:
        self.shutdown_requested = True
        return {"ok": True}

    # ------------------------------------------------------------------ media & projects

    def media_probe(self, params: dict[str, Any]) -> dict[str, Any]:
        return probe(_require(params, "path")).to_dict()

    def projects_create(self, params: dict[str, Any]) -> dict[str, Any]:
        media = probe(_require(params, "path"))
        title = params.get("title")
        if title is not None and not isinstance(title, str):
            raise InvalidParams("Parameter 'title' must be a string.")
        title = (title or "").strip() or Path(media.path).stem
        project = repo.create_project(self.db, media, title[:300])
        self.notify("project.updated", project)
        return project

    def projects_list(self, params: dict[str, Any]) -> dict[str, Any]:
        status = params.get("status")
        if status is not None and status not in repo.PROJECT_STATUSES:
            raise InvalidParams(f"Unknown status '{status}'.")
        return {"projects": repo.list_projects(self.db, status=status, limit=params.get("limit"))}

    def projects_get(self, params: dict[str, Any]) -> dict[str, Any]:
        return repo.get_project(self.db, _require(params, "project_id"))

    def projects_rename(self, params: dict[str, Any]) -> dict[str, Any]:
        title = _require(params, "title").strip()[:300]
        project = repo.rename_project(self.db, _require(params, "project_id"), title)
        self.notify("project.updated", project)
        return project

    def projects_delete(self, params: dict[str, Any]) -> dict[str, Any]:
        project_id = _require(params, "project_id")
        repo.get_project(self.db, project_id)  # 404 early
        self.jobs.cancel_project_jobs(project_id)
        job_ids = repo.delete_project(self.db, project_id)
        for job_id in job_ids:
            self.paths.remove_job_dir(job_id)
        self.notify("project.deleted", {"project_id": project_id})
        return {"deleted": project_id}

    # ------------------------------------------------------------------ transcription

    def transcription_preflight(self, params: dict[str, Any]) -> dict[str, Any]:
        options = normalize_options(params.get("options"))
        resolved = resolve(options, self.models)
        return {"options": options, "resolved": resolved.describe()}

    def jobs_start(self, params: dict[str, Any]) -> dict[str, Any]:
        project_id = _require(params, "project_id")
        options = normalize_options(params.get("options"))
        resolve(options, self.models)  # refuse to queue a job that cannot run
        return self.jobs.submit(project_id, options)

    def jobs_list(self, params: dict[str, Any]) -> dict[str, Any]:
        statuses = ["queued", "running"] if params.get("active_only") else params.get("statuses")
        return {"jobs": repo.list_jobs(self.db, statuses=statuses, project_id=params.get("project_id"),
                                       limit=int(params.get("limit", 100)))}

    def transcript_get(self, params: dict[str, Any]) -> dict[str, Any]:
        return repo.get_transcript(self.db, _require(params, "project_id"), include_words=bool(params.get("include_words")))

    def transcript_original(self, params: dict[str, Any]) -> dict[str, Any]:
        project_id = _require(params, "project_id")
        repo.get_project(self.db, project_id)
        original = repo.get_original_transcript(self.db, project_id)
        if original is None:
            raise NotFound("This project has no transcript yet.")
        return original

    def _edit(self, params: dict[str, Any], operation: Callable[[str], str | None]) -> dict[str, Any]:
        """Run a correction (or undo/redo) and return the updated transcript."""
        project_id = _require(params, "project_id")
        summary = operation(project_id)
        transcript = repo.get_transcript(self.db, project_id, include_words=True)
        if summary is not None:
            self.notify("transcript.updated", {"project_id": project_id, "revision": transcript["revision"],
                                               "summary": summary})
        return {"summary": summary, "changed": summary is not None, "transcript": transcript}

    # ------------------------------------------------------------------ export

    def _render(self, params: dict[str, Any]) -> tuple[str, str]:
        project_id = _require(params, "project_id")
        fmt = params.get("format", "txt")
        project = repo.get_project(self.db, project_id)
        transcript = repo.get_transcript(self.db, project_id, include_words=False)
        if not transcript["segments"]:
            raise NotFound("This project has no transcript yet.")
        return render(fmt, project, transcript, params.get("options") or {}), suggested_filename(project, fmt)

    def export_render(self, params: dict[str, Any]) -> dict[str, Any]:
        content, filename = self._render(params)
        return {"content": content, "suggested_filename": filename}

    def export_write(self, params: dict[str, Any]) -> dict[str, Any]:
        path = _require(params, "path")
        content, _ = self._render(params)
        written = write_atomic(path, content)
        return {"path": path, "bytes": written}

    # ------------------------------------------------------------------ models & settings

    def models_list(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {
            "models_dir": str(self.models.models_dir()),
            "models": self.models.list(),
            "downloads": self.downloads.snapshot(),
            "backends": describe_backends(),
        }

    def models_cancel_download(self, params: dict[str, Any]) -> dict[str, Any]:
        self.downloads.cancel(_require(params, "model_id"))
        return {"ok": True}

    def models_delete(self, params: dict[str, Any]) -> dict[str, Any]:
        spec = get_spec(_require(params, "model_id"))
        return {"model_id": spec.id, "status": self.models.delete(spec)}

    def settings_get(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {
            "models_dir": str(self.models.models_dir()),
            "default_models_dir": str(self.paths.default_models_dir),
        }

    def settings_set(self, params: dict[str, Any]) -> dict[str, Any]:
        if "models_dir" in params:
            value = params["models_dir"]
            if value is not None and not isinstance(value, str):
                raise InvalidParams("models_dir must be a string or null.")
            self.models.set_models_dir(value)
        return self.settings_get({})
