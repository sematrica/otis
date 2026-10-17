"""Job worker: runs one transcription job in its own process.

The worker persists every step to the database so a job can be resumed after
cancellation, a crash or the app quitting:

* extracted audio is reused if the source file is unchanged;
* the chunk plan is reused;
* each chunk's raw recognition result is stored as soon as it is produced,
  and completed chunks are never re-run on resume.

Events written to stdout (one JSON object per line) tell the server that the
job's persisted state changed; the server re-reads the database and notifies
the app.

Exit codes: 0 completed, 1 failed, 2 cancelled, 3 interrupted (parent gone).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from collections import Counter
from typing import Any, Callable

import numpy as np

from ..alignment.aligner import SegmentAlignment, WordAligner
from ..alignment.models import load_model as load_alignment_model
from ..asr.base import DecodeOptions, TranscriptionResult, shift_segments
from ..audio import SAMPLE_RATE
from ..asr.merge import apply_ownership, assemble
from ..asr.registry import create_backend
from ..audio.chunking import energy_envelope, plan_chunks
from ..audio.extract import extract_audio
from ..audio.wav import WavReader
from ..diarization.assign import assign_speakers
from ..diarization.clustering import SpeakerBounds
from ..diarization.pipeline import DiarizationResult, Diarizer
from ..errors import Cancelled, EngineError, SourceChanged
from ..media.probe import check_unchanged, probe
from ..models.manager import ModelManager
from ..paths import EnginePaths
from ..storage import repo
from ..storage.db import Database
from ..transcript.edit import TranscriptEditor
from ..transcript.build import apply_alignment, speakers_in_order
from ..transcript.model import TSegment
from ..util.hardware import performance_core_count
from ..verify.quality import SpeechMap, analyse, build_speech_map, missing_speech
from ..verify.recheck import Recheck, summary_text
from ..verify.retranscribe import load_decodes
from ..verify.validate import validate
from ..youtube import captions as youtube_captions
from ..youtube import fetch as youtube_fetch
from ..youtube import urls as youtube_urls
from .options import CHUNK_PAD_SECONDS, ResolvedJob, resolve

log = logging.getLogger(__name__)

EXIT_COMPLETED, EXIT_FAILED, EXIT_CANCELLED, EXIT_INTERRUPTED = 0, 1, 2, 3
PROMPT_WORDS = 48
#: Silence prepended to a chunk that starts at the very beginning of the
#: recording. Every later chunk begins inside a pause or an overlap, but a
#: recording can start mid-speech, and Whisper can drop a first word that
#: starts at sample 0 with no lead-in. Timestamps are shifted back afterwards.
LEAD_IN_SECONDS = 0.5
_PROGRESS_INTERVAL_S = 0.5


class _Interrupted(Exception):
    """Raised in the main thread when the server process has gone away."""


class Worker:
    def __init__(self, paths: EnginePaths, job_id: str, emit: Callable[[dict[str, Any]], None]):
        self.paths = paths
        self.job_id = job_id
        self.emit = emit
        self.db = Database(paths.db_path)
        self.models = ModelManager(self.db, paths)
        self.orphaned = False
        #: Set by signal handlers: "cancelled" or "interrupted". Any exception
        #: raised after a stop request is classified by it, because the signal
        #: can surface inside library code as a different exception type.
        self.stop_reason: str | None = None
        self._last_progress = 0.0
        self._last_cancel_poll = 0.0

    # ------------------------------------------------------------------ plumbing

    def _changed(self, **info: Any) -> None:
        self.emit({"event": "updated", "job_id": self.job_id, **info})

    def _check_cancel(self) -> None:
        row = self.db.query_one("SELECT cancel_requested FROM jobs WHERE id = ?", (self.job_id,))
        if row is not None and row["cancel_requested"]:
            raise Cancelled("The job was cancelled.")

    def _progress(self, stage: str, done: float, total: float | None, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_progress < _PROGRESS_INTERVAL_S:
            return
        self._last_progress = now
        repo.update_stage(self.db, self.job_id, stage, done=done, total=total)
        self._changed(stage=stage)

    def _start(self, stage: str, **kwargs: Any) -> None:
        repo.start_stage(self.db, self.job_id, stage, **kwargs)
        self._changed(stage=stage)

    def _finish(self, stage: str, status: str = "completed", detail: str | None = None) -> None:
        repo.finish_stage(self.db, self.job_id, stage, status, detail)
        self._changed(stage=stage)

    def _skip(self, stage: str, detail: str | None) -> None:
        repo.start_stage(self.db, self.job_id, stage, detail=detail)
        self._finish(stage, status="skipped", detail=detail)

    # ------------------------------------------------------------------ run

    def run(self) -> int:
        job = repo.get_job_row(self.db, self.job_id)
        project = repo.get_project(self.db, job["project_id"])
        options = json.loads(job["options_json"])
        try:
            if job["kind"] == "recheck":
                self._run_recheck(project, options)
            elif job["kind"] == "youtube":
                self._run_youtube(project, options)
            else:
                self._run(project, options)
        except Cancelled as exc:
            repo.finish_job(self.db, self.job_id, "cancelled", error_code=exc.code, error_message=exc.message)
            self._changed(status="cancelled")
            return EXIT_CANCELLED
        except _Interrupted:
            return self._finish_interrupted()
        except BaseException as exc:
            if self.stop_reason == "interrupted":
                return self._finish_interrupted()
            if self.stop_reason == "cancelled":
                repo.finish_job(self.db, self.job_id, "cancelled", error_code="cancelled", error_message="The job was cancelled.")
                self._changed(status="cancelled")
                return EXIT_CANCELLED
            if isinstance(exc, EngineError):
                log.warning("job %s failed: %s", self.job_id, exc.message)
                repo.finish_job(self.db, self.job_id, "failed", error_code=exc.code, error_message=exc.message)
                self._changed(status="failed")
                return EXIT_FAILED
            if not isinstance(exc, Exception):
                raise
            log.exception("job %s crashed", self.job_id)
            repo.finish_job(
                self.db, self.job_id, "failed", error_code="internal_error",
                error_message=f"Unexpected error: {type(exc).__name__}: {exc}",
            )
            self._changed(status="failed")
            return EXIT_FAILED
        repo.finish_job(self.db, self.job_id, "completed")
        self._changed(status="completed")
        return EXIT_COMPLETED

    def _finish_interrupted(self) -> int:
        repo.finish_job(
            self.db, self.job_id, "interrupted", error_code="interrupted",
            error_message="Processing stopped because the app quit. Resume to continue.",
        )
        return EXIT_INTERRUPTED

    def _run(self, project: dict[str, Any], options: dict[str, Any]) -> None:
        source = project["source"]
        media = project["media"]
        if media is None or not source["path"]:
            raise EngineError("This project's media has not been retrieved yet.", code="source_changed")

        # 1. prepare ----------------------------------------------------------
        self._start("prepare")
        if not check_unchanged(source["path"], source["size_bytes"], source["mtime_ns"]):
            raise SourceChanged(
                "The source file was moved, deleted or modified after it was imported. Import it again.",
                data={"path": source["path"]},
            )
        resolved = resolve(options, self.models)  # fails early if the model is missing
        self._finish("prepare")

        # 2. extract ------------------------------------------------------------
        job_dir = self.paths.job_dir(self.job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        wav_path = job_dir / "audio.wav"
        asset = repo.get_audio_asset(self.db, self.job_id)
        reusable = (
            asset is not None
            and wav_path.is_file()
            and asset["source_size"] == source["size_bytes"]
            and asset["source_mtime_ns"] == source["mtime_ns"]
        )
        if reusable:
            try:
                with WavReader(wav_path) as reader:
                    reusable = reader.samples == asset["samples"]
            except EngineError:
                reusable = False
        if reusable:
            self._start("extract", unit="seconds", total=asset["duration"], done=asset["duration"])
            self._finish("extract", detail="Reused audio extracted by an earlier attempt.")
        else:
            repo.delete_audio_asset(self.db, self.job_id)
            self.db.execute("DELETE FROM chunks WHERE job_id = ?", (self.job_id,))
            expected = media.get("duration")
            self._start("extract", unit="seconds", total=expected, done=0.0)
            result = extract_audio(
                source["path"],
                wav_path,
                stream_index=media.get("selected_audio_stream"),
                expected_duration=expected,
                progress=lambda done, total: self._progress("extract", done, total),
                should_cancel=self._poll_cancel,
            )
            report = {
                "decode_errors": result.decode_errors,
                "inserted_silence": round(result.inserted_silence, 3),
                "trimmed_overlap": round(result.trimmed_overlap, 3),
                "discontinuities": result.discontinuities,
            }
            repo.save_audio_asset(
                self.db, self.job_id, str(wav_path), result.sample_rate, result.samples, result.duration,
                source["size_bytes"], source["mtime_ns"], report,
            )
            repo.update_stage(self.db, self.job_id, "extract", done=result.duration, total=result.duration)
            notes = []
            if result.decode_errors:
                notes.append(f"{result.decode_errors} damaged packet(s) skipped")
            if result.discontinuities:
                notes.append(f"{result.discontinuities} timestamp discontinuity(ies)")
            self._finish("extract", detail="; ".join(notes) or None)

        reader = WavReader(wav_path)
        try:
            self._transcribe_all(project, reader, resolved)
        finally:
            reader.close()

    # ------------------------------------------------------------------ on-demand re-check

    def _run_youtube(self, project: dict[str, Any], options: dict[str, Any]) -> None:
        """Retrieve the media of a YouTube project (and the captions the user
        chose), then transcribe it like a local file. An interrupted retrieval
        continues where it stopped; finished media is never retrieved twice."""
        request = options.get("youtube") or {}
        target = youtube_urls.validate(project["source"]["url"] or "")  # checked again, here too
        meta = dict(project.get("source_meta") or {})
        source = project["source"]
        if source["path"] and check_unchanged(source["path"], source["size_bytes"], source["mtime_ns"]):
            repo.start_stage(self.db, self.job_id, "download")
            self._finish("download", detail="Reused the media retrieved earlier.")
        else:
            self._start("download", unit="bytes", total=None, done=0)
            info = youtube_fetch._extract(target, {"skip_download": True})
            refused = youtube_fetch.refusal(info)
            if refused is not None:
                raise EngineError(refused["message"], code=refused["code"])
            folder = self.paths.media_dir(project["id"])
            try:
                retrieved = youtube_fetch.retrieve(
                    target, info, folder, video=bool(request.get("video", True)),
                    progress=lambda done, total: self._progress("download", done, total),
                    should_cancel=self._poll_cancel)
            except youtube_fetch.Cancelled:
                self._check_cancel()
                raise Cancelled("The job was cancelled.") from None
            warnings = []
            tracks = []
            if request.get("captions"):
                try:
                    tracks = youtube_fetch.retrieve_captions(target, request["captions"], folder / "captions")
                    repo.save_captions(self.db, project["id"], tracks, source="youtube")
                except EngineError as exc:
                    warnings.append(f"Captions could not be retrieved: {exc.message}")
            media = probe(str(retrieved.path))
            meta["retrieved"] = {
                "file": retrieved.path.name, "bytes": retrieved.bytes, "formats": retrieved.formats,
                "has_video": media.has_video, "duration": media.duration,
                "captions": [{k: t[k] for k in ("language", "kind", "name")} for t in tracks],
                "warnings": warnings,
            }
            expected = meta.get("duration")
            if expected and media.duration and abs(media.duration - expected) > 2.0:
                meta["retrieved"]["warnings"].append(
                    f"The retrieved media lasts {media.duration:.0f} s; YouTube lists {expected:.0f} s.")
            repo.set_project_media(self.db, project["id"], media, meta)
            detail = f"{retrieved.bytes / 1e6:.1f} MB" + (f", {len(tracks)} caption track(s)" if tracks else "")
            self._finish("download", detail=detail)
        self._run(repo.get_project(self.db, project["id"]), options)

    def _source_marks(self, project: dict[str, Any], segments: list[TSegment], language: str | None,
                      effective: dict[str, Any]) -> None:
        """Marks that come from the source rather than the audio: where the
        published captions clearly differ, and the creator's sponsor chapters."""
        compared = []
        for track in repo.list_captions(self.db, project["id"], include_cues=True):
            if not youtube_captions.same_language(track["language"], language):
                continue
            cues = [youtube_captions.Cue(c["start"], c["end"], c["text"]) for c in track["cues"]]
            flags = youtube_captions.compare(segments, cues, kind=track["kind"], language=track["language"])
            for index, flag in flags.items():
                segments[index].flags.append(flag)
            compared.append({"language": track["language"], "kind": track["kind"], "passages_differ": len(flags)})
        chapters = youtube_fetch.sponsor_chapters({"chapters": [
            {"start_time": c.get("start"), "end_time": c.get("end"), "title": c.get("title")}
            for c in (project.get("source_meta") or {}).get("chapters") or []]})
        marked = 0
        for chapter in chapters:
            if chapter["start"] is None or chapter["end"] is None:
                continue
            for seg in segments:
                if seg.start < chapter["end"] and seg.end > chapter["start"]:
                    seg.flags.append({"type": "in_sponsor_chapter", "title": chapter["title"],
                                      "start": chapter["start"], "end": chapter["end"]})
                    marked += 1
        if compared or chapters:
            effective["source"] = {"captions_compared": compared, "sponsor_chapters": chapters,
                                   "segments_in_sponsor_chapters": marked}

    def _run_recheck(self, project: dict[str, Any], options: dict[str, Any]) -> None:
        """Re-check passages the user picked (passes 6-7) on the current
        transcript, with the audio of the transcription that produced it, and
        save the outcome as one undoable correction."""
        request = options.get("recheck_request") or {}
        source = project["source"]
        self._start("prepare")
        if not check_unchanged(source["path"], source["size_bytes"], source["mtime_ns"]):
            raise SourceChanged(
                "The source file was moved, deleted or modified after it was imported. Import it again.",
                data={"path": source["path"]},
            )
        wav_path = self.paths.job_dir(str(request.get("source_job", ""))) / "audio.wav"
        if not wav_path.is_file():
            raise EngineError("The audio extracted for this transcript is no longer available. Transcribe the "
                              "recording again to re-check passages.", code="audio_unavailable")
        resolved = resolve(options, self.models)
        rows = repo.load_segments(self.db, project["id"])
        ids = [sid for sid, _seg, _edited in rows]
        segments = [seg for _sid, seg, _edited in rows]
        wanted = set(request.get("segment_ids") or [])
        indices = {i for i, sid in enumerate(ids) if sid in wanted}
        protected = {i for i, (_sid, _seg, edited) in enumerate(rows) if edited}
        languages = Counter(seg.language for seg in segments if seg.language)
        language = languages.most_common(1)[0][0] if languages else None
        self._finish("prepare", detail=f"{len(indices)} passage(s) to re-check")

        reader = WavReader(wav_path)
        try:
            recheck = Recheck(segments, language, resolved, reader.duration, indices=indices, protected=protected)
            cache_dir = self.paths.job_dir(self.job_id)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / "recheck.jsonl"
            done = sum(1 for d in recheck.decodes if d[3] in load_decodes(cache_path))
            self._start("retranscribe", unit="decodes", total=len(recheck.decodes) or None, done=done,
                        detail=f"{len(recheck.targets)} passage(s)")
            recheck.decode(reader, cache_path,
                           progress=lambda d, total: self._progress("retranscribe", d, total, force=True),
                           check_cancel=self._check_cancel)
            speech = build_speech_map(reader, None, spans=[(w.start - 2.5, w.end + 2.5) for w in recheck.windows],
                                      should_cancel=self._poll_cancel)
            result = recheck.finish(reader, reanalyse=lambda changed: analyse(changed, speech, missing=False),
                                    speech=speech)
            self._finish("retranscribe", detail=summary_text(result.counts, result.info["acoustic_model"] is not None))
        finally:
            reader.close()

        self._check_cancel()
        self._start("finalize")
        touched = {t.segment for t in recheck.targets} | {i for i, _reason in recheck.skipped}
        updates: dict[int, dict[str, Any]] = {}
        for i in sorted(touched):
            seg = result.replaced.get(i, segments[i])
            words = None
            if i in result.replaced:
                words = [{"start": w.start, "end": w.end, "text": w.text, "probability": w.probability,
                          "timing_source": w.timing_source, "whisper_start": w.whisper_start,
                          "whisper_end": w.whisper_end, "alignment_score": w.alignment_score,
                          "original_text": w.text} for w in seg.words]
            updates[ids[i]] = {"text": seg.text, "start": seg.start, "end": seg.end, "flags": seg.flags, "words": words}
        n = len(touched)
        has_model = result.info["acoustic_model"] is not None
        summary = f"Re-check {n} passage{'s' if n != 1 else ''}: {summary_text(result.counts, has_model)}"
        TranscriptEditor(self.db).apply_recheck(project["id"], self.job_id, updates, summary)
        repo.set_effective(self.db, self.job_id, {"recheck": {"enabled": True, **result.info, **result.counts}})
        self._finish("finalize", detail=summary)

    def _poll_cancel(self) -> bool:
        # called frequently from decoding loops; DB check is cheap but throttle anyway
        now = time.monotonic()
        if now - self._last_cancel_poll < 0.5:
            return False
        self._last_cancel_poll = now
        row = self.db.query_one("SELECT cancel_requested FROM jobs WHERE id = ?", (self.job_id,))
        return bool(row and row["cancel_requested"])

    def _transcribe_all(self, project: dict[str, Any], reader: WavReader, resolved) -> None:
        duration = reader.duration

        # 3. analyze (chunk plan) -------------------------------------------
        existing = repo.load_chunks(self.db, self.job_id)
        if existing and abs(existing[-1][0].owned_end - duration) < 1e-3:
            self._start("analyze", unit="seconds", total=duration, done=duration)
            self._finish("analyze", detail=f"Reused plan: {len(existing)} chunk(s).")
        else:
            self._start("analyze", unit="seconds", total=duration, done=0.0)
            envelope = None
            if duration > resolved.chunk_seconds * 1.2:
                envelope = energy_envelope(
                    reader,
                    progress=lambda done, total: self._progress("analyze", done, total),
                    should_cancel=self._poll_cancel,
                )
            chunks = plan_chunks(duration, envelope, target_seconds=resolved.chunk_seconds, pad_seconds=CHUNK_PAD_SECONDS)
            repo.save_chunk_plan(self.db, self.job_id, chunks)
            repo.update_stage(self.db, self.job_id, "analyze", done=duration, total=duration)
            in_speech = sum(1 for c in chunks if c.cut_in_speech)
            detail = f"{len(chunks)} chunk(s)"
            if in_speech:
                detail += f"; {in_speech} cut(s) without a clear pause (overlap-protected)"
            self._finish("analyze", detail=detail)
            existing = repo.load_chunks(self.db, self.job_id)

        # 4. load model -----------------------------------------------------
        self._check_cancel()
        self._start("load_model", detail=resolved.model["display_name"])
        backend = create_backend(resolved.backend)
        load_settings = backend.load(resolved.model_dir, resolved.load)
        effective: dict[str, Any] = {
            **resolved.describe(),
            "load": load_settings,
            "chunking": {"chunks": len(existing), "target_seconds": resolved.chunk_seconds, "pad_seconds": CHUNK_PAD_SECONDS},
        }
        repo.set_effective(self.db, self.job_id, effective)
        self._finish("load_model", detail=f"{resolved.model['display_name']} on {backend.display_name}")

        # 5. transcribe ---------------------------------------------------------
        owned_total = sum(c.owned_end - c.owned_start for c, _ in existing)
        done_base = sum(c.owned_end - c.owned_start for c, row in existing if row["status"] == "completed")
        pending = [c for c, row in existing if row["status"] != "completed"]
        self._start("transcribe", unit="seconds", total=owned_total, done=done_base,
                    detail=f"{len(existing) - len(pending)} of {len(existing)} chunk(s) already done" if done_base else None)
        results: dict[int, TranscriptionResult] = {
            c.index: TranscriptionResult.from_dict(json.loads(row["result_json"]))
            for c, row in existing
            if row["status"] == "completed"
        }
        decode_effective = None
        for chunk in pending:
            self._check_cancel()
            prompt = self._prompt_from_previous(chunk.index, existing, results)
            options = DecodeOptions(**{**resolved.decode.__dict__, "initial_prompt": prompt})
            audio = reader.read(chunk.audio_start, chunk.audio_end)
            lead_in = LEAD_IN_SECONDS if chunk.audio_start <= 0.0 else 0.0
            if lead_in:
                audio = np.concatenate([np.zeros(int(lead_in * SAMPLE_RATE), dtype=np.float32), audio])
            span = chunk.owned_end - chunk.owned_start
            started = time.monotonic()

            def on_progress(fraction: float, base=done_base, span=span) -> None:
                self._progress("transcribe", base + span * fraction, owned_total)

            result, decode_effective = backend.transcribe(audio, options, on_progress)
            if lead_in:
                result.segments = shift_segments(result.segments, -lead_in)
                result.duration -= lead_in
            repo.complete_chunk(self.db, self.job_id, chunk.index, result.to_dict(), time.monotonic() - started)
            results[chunk.index] = result
            done_base += span
            self._progress("transcribe", done_base, owned_total, force=True)
            repo.update_stage(self.db, self.job_id, "transcribe", detail=f"{len(results)} of {len(existing)} chunk(s) done")
        if decode_effective is not None:
            effective["decode"] = {k: v for k, v in decode_effective.items() if k not in load_settings and k != "backend"}
            repo.set_effective(self.db, self.job_id, effective)
        # Free the recogniser before the speaker and alignment models load:
        # large models never share memory.
        backend.close()
        del backend
        self._finish("transcribe")

        pairs = [(c, results[c.index]) for c, _ in existing]
        merged = assemble(pairs, pad=CHUNK_PAD_SECONDS)
        language = _dominant_language(pairs)
        segments = [TSegment.from_merged(m) for m in merged]

        # 6. speakers -------------------------------------------------------------
        diarization = self._diarize(reader, resolved, effective)
        repo.set_effective(self.db, self.job_id, effective)

        # 7. word alignment -------------------------------------------------------
        self._align(reader, segments, language, resolved, effective)
        repo.set_effective(self.db, self.job_id, effective)
        if diarization is not None:  # a speaker for every word (needed by the quality checks)
            segments = assign_speakers(segments, diarization)

        # 8. quality analysis (pass 5) ------------------------------------------------
        speech = self._quality(reader, segments, diarization, effective)
        repo.set_effective(self.db, self.job_id, effective)

        # 9. targeted re-transcription, compared by acoustic evidence (passes 6-7) ----
        # (with an empty transcript, the language Whisper detected is all there is)
        heard = language or _detected_language(pairs)
        segments = self._recheck(reader, segments, heard, resolved, diarization, speech, effective)
        if language is None and segments:
            language = heard
        repo.set_effective(self.db, self.job_id, effective)

        # 10. validation (pass 8) -----------------------------------------------------
        self._check_cancel()
        self._start("validate")
        segments, validation = validate(segments, speech, reader.duration)
        effective["validation"] = validation
        repo.set_effective(self.db, self.job_id, effective)
        detail = f"{validation['needs_review']} passage(s) to review"
        if validation["missing_speech"] and not segments:
            detail = f"Speech detected in {validation['missing_speech']} place(s) but no words recognised"
        if validation["timing_fixes"]:
            detail += f"; {validation['timing_fixes']} timestamp(s) corrected"
        self._finish("validate", detail=detail)

        # 11. finalize --------------------------------------------------------------
        self._check_cancel()
        self._start("finalize")
        if project["source"]["kind"] == "url":
            self._source_marks(project, segments, language, effective)
            repo.set_effective(self.db, self.job_id, effective)
        speakers = speakers_in_order(segments) if diarization is not None else []
        if diarization is not None:
            effective["diarization"]["speakers_in_transcript"] = len(speakers)
            repo.set_effective(self.db, self.job_id, effective)
        count = repo.replace_transcript(self.db, project["id"], self.job_id, segments, language, speakers)
        removed = sum(1 for s in merged for f in s.flags if f.get("type") == "boundary_duplicate_removed")
        detail = f"{count} segment(s)"
        if speakers:
            detail += f", {len(speakers)} speaker(s)"
        if removed:
            detail += f"; {removed} duplicate(s) at chunk boundaries removed (recorded in flags)"
        self._finish("finalize", detail=detail)

    # ------------------------------------------------------------------ speakers

    def _diarize(self, reader: WavReader, resolved: ResolvedJob, effective: dict[str, Any]) -> DiarizationResult | None:
        if resolved.diarization_dir is None:
            effective["diarization"] = {"enabled": False, "reason": resolved.diarization_skip_reason}
            self._skip("diarize", resolved.diarization_skip_reason)
            return None
        self._check_cancel()
        settings = {
            "model": str(resolved.diarization_dir),
            "speakers": {k: v for k, v in resolved.speaker_bounds.items() if v is not None},
            "version": DIARIZATION_VERSION,
        }
        path = self.paths.job_dir(self.job_id) / "diarization.npz"
        result = None
        if path.is_file():
            try:
                cached = DiarizationResult.load(path)
                if cached.info.get("settings") == settings and abs(cached.duration - reader.duration) < 1e-3:
                    result = cached
            except (OSError, ValueError, KeyError):
                result = None
        duration = reader.duration
        if result is not None:
            self._start("diarize", unit="seconds", total=duration, done=duration)
            reused = True
        else:
            reused = False
            self._start("diarize", unit="seconds", total=duration, done=0.0)
            diarizer = Diarizer(
                resolved.diarization_dir / "segmentation.onnx",
                resolved.diarization_dir / "embedding.onnx",
                threads=performance_core_count(),
            )
            bounds = resolved.speaker_bounds
            result = diarizer.run(
                reader,
                SpeakerBounds(num_speakers=bounds.get("num"), min_speakers=bounds.get("min"), max_speakers=bounds.get("max")),
                progress=lambda fraction: self._progress("diarize", fraction * duration, duration),
                should_cancel=self._poll_cancel,
            )
            result.info["settings"] = settings
            temporary = path.with_name("diarization.partial.npz")
            result.save(temporary)
            os.replace(temporary, path)
            repo.update_stage(self.db, self.job_id, "diarize", done=duration, total=duration)
        overlap = sum(b - a for a, b in result.overlaps())
        effective["diarization"] = {
            "enabled": True,
            "method": "pyannote segmentation-3.0 + WeSpeaker ResNet34 + agglomerative clustering (ONNX Runtime)",
            "speakers_found": result.num_speakers,
            "overlap_seconds": round(overlap, 2),
            **{k: v for k, v in result.info.items() if k != "settings"},
        }
        detail = f"{result.num_speakers} speaker(s) found"
        if overlap >= 0.5:
            detail += f"; {overlap:.0f} s of overlapping speech"
        if reused:
            detail += " (reused from an earlier attempt)"
        self._finish("diarize", detail=detail)
        return result

    # ------------------------------------------------------------------ quality (pass 5)

    def _quality(self, reader: WavReader, segments: list[TSegment], diarization: DiarizationResult | None,
                 effective: dict[str, Any]) -> SpeechMap:
        self._check_cancel()
        duration = reader.duration
        self._start("quality", unit="seconds", total=duration, done=0.0)
        speech = build_speech_map(
            reader, diarization,
            progress=lambda done, total: self._progress("quality", done, total),
            should_cancel=self._poll_cancel,
        )
        counts = analyse(segments, speech)
        effective["quality"] = {
            "speech_detectors": ["silero-vad"] + (["diarization"] if diarization is not None else []),
            "flags": counts,
        }
        labels = {"low_confidence": "unclear", "possible_hallucination": "possible hallucination(s)",
                  "missing_speech": "untranscribed speech", "repeated_words": "repetition(s)"}
        found = [f"{counts[k]} {v}" for k, v in labels.items() if counts.get(k)]
        self._finish("quality", detail="; ".join(found) or "No doubtful passages found.")
        return speech

    # ------------------------------------------------------------------ re-check (passes 6-7)

    def _recheck(self, reader: WavReader, segments: list[TSegment], language: str | None, resolved: ResolvedJob,
                 diarization: DiarizationResult | None, speech: SpeechMap, effective: dict[str, Any]) -> list[TSegment]:
        if not resolved.recheck:
            effective["recheck"] = {"enabled": False, "reason": resolved.recheck_skip_reason}
            self._skip("retranscribe", resolved.recheck_skip_reason)
            return segments
        self._check_cancel()
        gaps = missing_speech([], speech) if not segments else ()
        recheck = Recheck(segments, language, resolved, reader.duration, gaps=gaps)
        if not recheck.targets:
            effective["recheck"] = {"enabled": True, "targets": 0}
            self._start("retranscribe")
            self._finish("retranscribe", detail="No doubtful passages to re-check.")
            return segments
        cache_path = self.paths.job_dir(self.job_id) / "recheck.jsonl"
        done = sum(1 for d in recheck.decodes if d[3] in load_decodes(cache_path))
        self._start("retranscribe", unit="decodes", total=len(recheck.decodes), done=done,
                    detail=f"{len(recheck.targets)} passage(s) in {len(recheck.windows)} window(s)")
        recheck.decode(reader, cache_path,
                       progress=lambda d, total: self._progress("retranscribe", d, total, force=True),
                       check_cancel=self._check_cancel)
        result = recheck.finish(
            reader,
            assign=(lambda seg: assign_speakers([seg], diarization)) if diarization is not None else None,
            reanalyse=lambda changed: analyse(changed, speech, missing=False),
            speech=speech,
        )
        effective["recheck"] = {"enabled": True, **result.info, **result.counts}
        self._finish("retranscribe", detail=summary_text(result.counts, result.info["acoustic_model"] is not None))
        return result.segments

    # ------------------------------------------------------------------ alignment

    def _align(self, reader: WavReader, segments: list[TSegment], language: str | None, resolved: ResolvedJob,
               effective: dict[str, Any]) -> None:
        if not resolved.alignment_candidates:
            effective["alignment"] = {"enabled": False, "reason": resolved.alignment_skip_reason}
            self._skip("align", resolved.alignment_skip_reason)
            return
        choice = resolved.alignment_model_for(language)
        if choice is None:
            reason = f"No installed word-alignment model supports this language ({language or 'unknown'}); word times come from Whisper."
            effective["alignment"] = {"enabled": False, "reason": reason}
            self._skip("align", reason)
            return
        spec, model_dir = choice
        supported = [i for i, seg in enumerate(segments)
                     if seg.words and (not spec.languages or (seg.language or language) in spec.languages)]
        targets = [segments[i] for i in supported]
        span_total = sum(max(0.0, s.end - s.start) for s in targets) or 1.0
        self._check_cancel()
        self._start("align", unit="seconds", total=span_total, done=0.0, detail=spec.display_name)

        # Results are appended per group, so an interrupted job resumes here.
        key = _alignment_key(spec.id, targets)
        progress_path = self.paths.job_dir(self.job_id) / "alignment.jsonl"
        done: dict[int, SegmentAlignment] = {}
        if progress_path.is_file():
            for line in progress_path.read_text().splitlines():
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if entry.get("key") == key:
                    done.update({int(i): SegmentAlignment.from_dict(r) for i, r in entry["results"].items()})
        else:
            progress_path.touch()

        def on_group(_group, new: dict[int, SegmentAlignment]) -> None:
            with open(progress_path, "a") as f:
                f.write(json.dumps({"key": key, "results": {str(i): r.to_dict() for i, r in new.items()}}) + "\n")

        model = load_alignment_model(spec, model_dir, performance_core_count())
        aligner = WordAligner(model)
        results = aligner.align(
            reader, targets, language, done=done, on_group=on_group,
            progress=lambda processed, total: self._progress("align", processed / total * span_total, span_total),
            should_cancel=self._poll_cancel,
        )
        del model, aligner
        by_segment: list[SegmentAlignment | None] = [None] * len(segments)
        for i, res in zip(supported, results):
            by_segment[i] = res
        summary = apply_alignment(segments, by_segment)
        effective["alignment"] = {"enabled": True, "model": {"id": spec.id, "display_name": spec.display_name}, **summary}
        detail = f"{summary['aligned']} of {summary['segments']} segment(s) aligned"
        if summary["failed"]:
            detail += f"; {summary['failed']} kept Whisper's timing (flagged)"
        self._finish("align", detail=detail)

    @staticmethod
    def _prompt_from_previous(index: int, chunks, results: dict[int, TranscriptionResult]) -> str | None:
        """Text context from the end of the previous chunk (what Whisper itself
        does between its 30-second windows)."""
        if index == 0 or (index - 1) not in results:
            return None
        previous_chunk = next(c for c, _ in chunks if c.index == index - 1)
        owned = apply_ownership(previous_chunk, results[index - 1], is_last=False)
        words = " ".join(s.text for s in owned).split()
        return " ".join(words[-PROMPT_WORDS:]) or None


#: Bump when diarization output changes meaning, so cached results are redone.
DIARIZATION_VERSION = 1


def _alignment_key(model_id: str, segments: list[TSegment]) -> str:
    import hashlib

    digest = hashlib.sha256(model_id.encode())
    for seg in segments:
        digest.update(f"{seg.start:.3f}|{seg.end:.3f}|{seg.text}|{len(seg.words)}\n".encode())
    return digest.hexdigest()[:16]


def _detected_language(pairs) -> str | None:
    """Language Whisper detected over most of the recording, words or not."""
    weights: Counter[str] = Counter()
    for chunk, result in pairs:
        if result.language:
            weights[result.language] += chunk.owned_end - chunk.owned_start
    return weights.most_common(1)[0][0] if weights else None


def _dominant_language(pairs) -> str | None:
    weights: Counter[str] = Counter()
    for chunk, result in pairs:
        if result.language and result.segments:
            weights[result.language] += chunk.owned_end - chunk.owned_start
    return weights.most_common(1)[0][0] if weights else None


# ---------------------------------------------------------------------- entry point


def main(job_id: str, data_dir: str | None) -> int:
    paths = EnginePaths.create(data_dir)
    out = sys.stdout
    lock = threading.Lock()

    def emit(event: dict[str, Any]) -> None:
        with lock:
            try:
                out.write(json.dumps(event) + "\n")
                out.flush()
            except (BrokenPipeError, ValueError):
                pass

    worker = Worker(paths, job_id, emit)
    original_parent = os.getppid()

    # SIGTERM = the user cancelled the job; SIGINT = processing was interrupted
    # (the app quit, the server disappeared, or Ctrl-C) and should be resumable.
    def on_term(_signum, _frame):
        if worker.stop_reason is None:
            worker.stop_reason = "cancelled"
            raise Cancelled("The job was cancelled.")

    def on_int(_signum, _frame):
        if worker.stop_reason is None:
            worker.stop_reason = "interrupted"
            raise _Interrupted()

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_int)

    def watchdog() -> None:
        while True:
            time.sleep(1.0)
            if os.getppid() != original_parent:
                worker.orphaned = True
                log.warning("server process went away; stopping job %s", job_id)
                os.kill(os.getpid(), signal.SIGINT)
                return

    threading.Thread(target=watchdog, daemon=True, name="parent-watchdog").start()
    return worker.run()


def default_env() -> dict[str, str]:
    """Environment for worker processes. Models are never fetched by a job;
    only a YouTube import job uses the network, for the video the user chose."""
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env
