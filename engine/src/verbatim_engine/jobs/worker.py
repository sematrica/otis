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

from ..asr.base import DecodeOptions, TranscriptionResult, shift_segments
from ..audio import SAMPLE_RATE
from ..asr.merge import apply_ownership, assemble
from ..asr.registry import create_backend
from ..audio.chunking import energy_envelope, plan_chunks
from ..audio.extract import extract_audio
from ..audio.wav import WavReader
from ..errors import Cancelled, EngineError, SourceChanged
from ..media.probe import check_unchanged
from ..models.manager import ModelManager
from ..paths import EnginePaths
from ..storage import repo
from ..storage.db import Database
from .options import CHUNK_PAD_SECONDS, resolve

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

    # ------------------------------------------------------------------ run

    def run(self) -> int:
        job = repo.get_job_row(self.db, self.job_id)
        project = repo.get_project(self.db, job["project_id"])
        options = json.loads(job["options_json"])
        try:
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
        backend.close()
        self._finish("transcribe")

        # 6. finalize ---------------------------------------------------------------
        self._check_cancel()
        self._start("finalize")
        pairs = [(c, results[c.index]) for c, _ in existing]
        merged = assemble(pairs, pad=CHUNK_PAD_SECONDS)
        language = _dominant_language(pairs)
        count = repo.replace_transcript(self.db, project["id"], self.job_id, merged, language)
        removed = sum(1 for s in merged for f in s.flags if f.get("type") == "boundary_duplicate_removed")
        detail = f"{count} segment(s)"
        if removed:
            detail += f"; {removed} duplicate(s) at chunk boundaries removed (recorded in flags)"
        self._finish("finalize", detail=detail)

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
    """Environment for worker processes: never touch the network."""
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env
