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
from ..errors import EngineError, InvalidParams, NotFound
from ..search import search as search_transcripts
from ..export.subtitles import render_captions
from ..youtube import fetch as youtube_fetch
from ..youtube import urls as youtube_urls
from ..export import FORMATS, WORD_FORMATS, render, suggested_filename, write_atomic
from ..jobs.manager import JobManager
from ..llm.providers import make_provider
from ..llm.runs import ReviewManager, decide, get_run, get_suggestion, list_runs, list_suggestions
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
from ..util.ids import utc_now
from ..util.timecode import format_clock

log = logging.getLogger(__name__)

Notify = Callable[[str, dict[str, Any]], None]


def _require(params: dict[str, Any], key: str, kind: type = str) -> Any:
    value = params.get(key)
    if value is None or (kind is str and not str(value).strip()):
        raise InvalidParams(f"Missing required parameter '{key}'.")
    if not isinstance(value, kind):
        raise InvalidParams(f"Parameter '{key}' must be of type {kind.__name__}.")
    return value


def _str_list(params: dict[str, Any], key: str) -> list[str]:
    value = params.get(key)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise InvalidParams(f"Parameter '{key}' must be a list of strings.")
    return value


def _int_list(params: dict[str, Any], key: str) -> list[int]:
    value = params.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        raise InvalidParams(f"Parameter '{key}' must be a non-empty list of integers.")
    return value


#: What the user confirms before a YouTube video is retrieved (stored with the project).
AUTHORIZATION_STATEMENT = ("I am authorized to download and transcribe this video: it is my own, it is licensed "
                           "for reuse, or I have the rights holder's permission.")


class EngineService:
    def __init__(self, paths: EnginePaths, notify: Notify, worker_command: list[str] | None = None):
        self.paths = paths
        self.notify = notify
        self.db = Database(paths.db_path)
        self.models = ModelManager(self.db, paths)
        self.jobs = JobManager(self.db, paths, notify, worker_command)
        self.downloads = DownloadManager(paths, notify, worker_command)
        self.editor = TranscriptEditor(self.db)
        self.reviews = ReviewManager(self.db, notify)
        self.shutdown_requested = False

    def start(self) -> None:
        self.jobs.start()

    def shutdown(self) -> None:
        self.reviews.shutdown()
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
            "transcript.deleteSegment": lambda p: self._edit(p, lambda pid: self.editor.delete_segment(
                pid, _require(p, "segment_id", int))),
            "transcript.chooseWording": lambda p: self._edit(p, lambda pid: self.editor.choose_wording(
                pid, _require(p, "segment_id", int), _require(p, "source"))),
            "transcript.markReviewed": lambda p: self._edit(p, lambda pid: self.editor.mark_reviewed(
                pid, _int_list(p, "segment_ids"), bool(p.get("reviewed", True)))),
            "transcript.recheck": lambda p: self.jobs.submit_recheck(_require(p, "project_id"),
                                                                     _int_list(p, "segment_ids")),
            "transcript.undo": lambda p: self._edit(p, self.editor.undo),
            "transcript.redo": lambda p: self._edit(p, self.editor.redo),
            "youtube.inspect": self.youtube_inspect,
            "projects.createFromYouTube": self.projects_create_from_youtube,
            "captions.list": lambda p: {"captions": repo.list_captions(
                self.db, _require(p, "project_id"), include_cues=bool(p.get("include_cues")))},
            "captions.export": self.captions_export,
            "queue.get": lambda _p: self.jobs.queue_state(),
            "queue.pause": lambda p: self.jobs.pause(interrupt=bool(p.get("interrupt"))),
            "queue.resume": lambda _p: self.jobs.resume_queue(),
            "queue.reorder": lambda p: self.jobs.reorder(_str_list(p, "job_ids")),
            "queue.setAutoResume": lambda p: self.jobs.set_auto_resume(_require(p, "enabled", bool)),
            "search.transcripts": lambda p: search_transcripts(
                self.db, _require(p, "query"), project_id=p.get("project_id"), speaker_id=p.get("speaker_id"),
                include_original=bool(p.get("include_original")), limit=p.get("limit") or 100),
            "export.formats": lambda _p: {"formats": list(FORMATS)},
            "export.render": self.export_render,
            "export.write": self.export_write,
            "models.list": self.models_list,
            "models.download": lambda p: self.downloads.start(_require(p, "model_id")),
            "models.cancelDownload": self.models_cancel_download,
            "models.delete": self.models_delete,
            "llm.models": self.llm_models,
            "llm.review": self.llm_review,
            "llm.runs": lambda p: {"runs": list_runs(self.db, _require(p, "project_id"))},
            "llm.run": lambda p: get_run(self.db, _require(p, "run_id", int), bool(p.get("exchanges"))),
            "llm.cancel": lambda p: self.reviews.cancel(_require(p, "run_id", int)),
            "suggestions.list": lambda p: {"suggestions": list_suggestions(
                self.db, _require(p, "project_id"), p.get("statuses"))},
            "suggestions.accept": self.suggestions_accept,
            "suggestions.reject": lambda p: decide(self.db, _require(p, "suggestion_id", int), "rejected"),
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
        self.paths.remove_media_dir(project_id)  # media the engine retrieved (YouTube)
        self.notify("project.deleted", {"project_id": project_id})
        return {"deleted": project_id}

    # ------------------------------------------------------------------ YouTube

    def youtube_inspect(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate a link and read the video's metadata (nothing is downloaded)."""
        return youtube_fetch.inspect(youtube_urls.validate(_require(params, "url")))

    def projects_create_from_youtube(self, params: dict[str, Any]) -> dict[str, Any]:
        """Create a project for a YouTube video and queue its retrieval and
        transcription. ``authorized`` must be true: the user confirms they may
        download this video (their own, licensed for reuse, or with permission)."""
        if params.get("authorized") is not True:
            raise InvalidParams("Confirm that you are authorized to download and transcribe this video.",
                                data={"reason": "authorization"})
        target = youtube_urls.validate(_require(params, "url"))
        options = normalize_options(params.get("options"))
        resolve(options, self.models)  # refuse before downloading anything if the model is missing
        inspected = youtube_fetch.inspect(target)
        if inspected["refusal"] is not None:
            raise EngineError(inspected["refusal"]["message"], code=inspected["refusal"]["code"])
        wanted = params.get("captions") or []
        if not isinstance(wanted, list):
            raise InvalidParams("Parameter 'captions' must be a list.")
        available = {(t["language"], t["kind"]): t for t in inspected["captions"]}
        tracks = []
        for item in wanted:
            key = (item.get("language"), item.get("kind")) if isinstance(item, dict) else None
            if key not in available:
                raise InvalidParams("That caption track is not offered for this video.", data={"track": item})
            tracks.append(available[key])
        video = params.get("media", "video") != "audio"
        meta = {
            "kind": "youtube", **{k: inspected[k] for k in ("url", "video_id", "title", "channel", "channel_url",
                                                           "duration", "upload_date", "license", "language",
                                                           "chapters", "sponsor_chapters", "ads_note")},
            "authorization": {"confirmed_at": utc_now(), "statement": AUTHORIZATION_STATEMENT},
            "requested": {"media": "video" if video else "audio", "captions": [
                {k: t[k] for k in ("language", "kind")} for t in tracks]},
        }
        title = (params.get("title") or inspected["title"] or target.video_id or "YouTube video").strip()[:300]
        project = repo.create_url_project(self.db, target.url, title, meta, has_video=video)
        job_options = {**options, "youtube": {"video": video, "captions": tracks}}
        job = repo.create_job(self.db, project["id"], options["mode"], job_options, kind="youtube")
        self.jobs.announce_new(job["id"])
        project = repo.get_project(self.db, project["id"])
        self.notify("project.updated", project)
        return {"project": project, "job": job}

    def captions_export(self, params: dict[str, Any]) -> dict[str, Any]:
        """A published caption track as SRT or WebVTT, as published (it is not
        the transcript)."""
        project_id = _require(params, "project_id")
        caption_id = _require(params, "caption_id", int)
        fmt = params.get("format", "vtt")
        if fmt not in ("srt", "vtt"):
            raise InvalidParams("Captions can be exported as 'srt' or 'vtt'.")
        track = next((t for t in repo.list_captions(self.db, project_id, include_cues=True) if t["id"] == caption_id),
                     None)
        if track is None:
            raise NotFound("Caption track not found.", data={"caption_id": caption_id})
        content = render_captions(track["cues"], fmt)
        project = repo.get_project(self.db, project_id)
        filename = suggested_filename(project, "txt").removesuffix(".txt") + f".{track['language']}.captions.{fmt}"
        if params.get("path"):
            return {"path": params["path"], "bytes": write_atomic(params["path"], content)}
        return {"content": content, "suggested_filename": filename}

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

    # ------------------------------------------------------------------ AI verification

    @staticmethod
    def _provider(params: dict[str, Any], require_model: bool = True):
        return make_provider(_require(params, "provider"), params.get("model"), base_url=params.get("base_url"),
                             api_key=params.get("api_key"), require_model=require_model)

    def llm_models(self, params: dict[str, Any]) -> dict[str, Any]:
        """Models offered by a provider (nothing from any transcript is sent)."""
        provider = self._provider(params, require_model=False)
        return {"provider": provider.name, "endpoint": provider.base_url, "remote": provider.remote,
                "models": provider.list_models()}

    def llm_review(self, params: dict[str, Any]) -> dict[str, Any]:
        provider = self._provider(params)
        return self.reviews.start(_require(params, "project_id"), provider,
                                  allow_remote=params.get("allow_remote") is True)

    def suggestions_accept(self, params: dict[str, Any]) -> dict[str, Any]:
        """Apply one AI suggestion through the editor (undoable), or start an
        acoustic re-check for a re-transcribe suggestion."""
        suggestion = get_suggestion(self.db, _require(params, "suggestion_id", int))
        if suggestion["status"] != "pending":
            raise InvalidParams(f"This suggestion is already {suggestion['status']}.")
        project_id, segment_id = suggestion["project_id"], suggestion["segment_id"]
        row = self.db.query_one("SELECT text, start FROM segments WHERE id = ? AND project_id = ?",
                                (segment_id, project_id)) if segment_id is not None else None
        if row is None or row["text"].strip() != suggestion["original_text"].strip():
            outdated = decide(self.db, suggestion["id"], "outdated")
            return {"suggestion": outdated, "summary": None, "changed": False, "job": None}
        run = get_run(self.db, suggestion["run_id"])
        source = f"{run['provider']} / {run['model']}"
        kind, clock = suggestion["kind"], format_clock(row["start"])
        job = None
        if kind in ("punctuation", "terminology"):
            result = self._edit({"project_id": project_id}, lambda pid: self.editor.edit_text(
                pid, segment_id, suggestion["suggested_text"], kind="ai_suggestion",
                summary=f"Apply AI {kind} suggestion at {clock} ({source})"))
        elif kind == "suspicious":
            note = f"AI review ({source}): {suggestion['reason']}"
            result = self._edit({"project_id": project_id}, lambda pid: self.editor.mark_uncertain(
                pid, segment_id, True, note))
        else:  # retranscribe: the audio decides, not the language model
            job = self.jobs.submit_recheck(project_id, [segment_id])
            result = {"summary": None, "changed": False}
        accepted = decide(self.db, suggestion["id"], "accepted")
        return {**result, "suggestion": accepted, "job": job}

    # ------------------------------------------------------------------ export

    def _render(self, params: dict[str, Any]) -> tuple[str, str]:
        project_id = _require(params, "project_id")
        fmt = params.get("format", "txt")
        project = repo.get_project(self.db, project_id)
        transcript = repo.get_transcript(self.db, project_id, include_words=fmt in WORD_FORMATS)
        if not transcript["segments"]:
            raise NotFound("This project has no transcript yet.")
        options = dict(params.get("options") or {})
        if fmt == "json":
            options["_original"] = repo.get_original_transcript(self.db, project_id)
            if transcript.get("job_id"):
                transcript["_job"] = repo.get_job(self.db, transcript["job_id"])
        return render(fmt, project, transcript, options), suggested_filename(project, fmt)

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
