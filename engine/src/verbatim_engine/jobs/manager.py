"""Persistent Job Manager (server side).

Jobs are queued in the database and run one at a time, in queue order — on a
16 GB machine two large models must never be loaded together. Each job runs
in a worker subprocess; this manager spawns it, relays its progress, enforces
cancellation, and reconciles the job's final state when the worker exits for
any reason (including native crashes).

The queue is persistent: its order and its paused state are stored, and work
interrupted by quitting (or a crash of the engine) is queued again at the
front when the engine starts, unless the user turned that off. A job is only
resumed automatically while it is its project's latest job, and at most
:data:`MAX_AUTO_ATTEMPTS` times.
"""

from __future__ import annotations

import json
import logging
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ..errors import InvalidJobState, NotFound
from ..paths import EnginePaths
from ..storage import repo
from ..storage.db import Database
from .options import normalize_options
from .worker import default_env

log = logging.getLogger(__name__)

Notify = Callable[[str, dict[str, Any]], None]

CANCEL_GRACE_SECONDS = 10.0
#: Attempts after which interrupted work is no longer resumed automatically.
MAX_AUTO_ATTEMPTS = 5
SETTING_PAUSED = "queue.paused"
SETTING_AUTO_RESUME = "queue.auto_resume"


class _Running:
    def __init__(self, job_id: str, process: subprocess.Popen, log_path: Path):
        self.job_id = job_id
        self.process = process
        self.log_path = log_path
        self.cancel_requested = False
        self.shutdown_requested = False
        #: stopped by "pause now": queued again at the front afterwards
        self.pause_requested = False
        self.kill_timer: threading.Timer | None = None
        self.monitor: threading.Thread | None = None


class JobManager:
    def __init__(self, db: Database, paths: EnginePaths, notify: Notify, worker_command: list[str] | None = None):
        self.db = db
        self.paths = paths
        self.notify = notify
        self.worker_command = worker_command or [sys.executable, "-m", "verbatim_engine"]
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._running: _Running | None = None
        self._scheduler: threading.Thread | None = None

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        self.recover()
        if repo.get_setting(self.db, SETTING_AUTO_RESUME, True):
            self.resume_interrupted()
        self._scheduler = threading.Thread(target=self._schedule_loop, name="job-scheduler", daemon=True)
        self._scheduler.start()

    def recover(self) -> list[str]:
        """Jobs left 'running' by a previous engine process cannot still be
        making progress (workers exit when their server disappears)."""
        recovered = []
        for job in repo.list_jobs(self.db, statuses=["running"], limit=1000):
            repo.finish_job(
                self.db,
                job["id"],
                "interrupted",
                error_code="interrupted",
                error_message="Processing stopped because the app quit or crashed. Resume to continue.",
            )
            recovered.append(job["id"])
        if recovered:
            log.info("marked %d job(s) as interrupted", len(recovered))
        return recovered

    def resume_interrupted(self) -> list[str]:
        """Queue interrupted work again, at the front, in its queue order."""
        resumed = []
        rows = self.db.query(
            """SELECT j.id, j.project_id, j.attempt FROM jobs j JOIN projects p ON p.id = j.project_id
               WHERE j.status = 'interrupted' AND p.last_job_id = j.id
               ORDER BY j.queue_position DESC, j.created_at DESC""")
        for row in rows:  # last first: each goes to the front
            if row["attempt"] >= MAX_AUTO_ATTEMPTS:
                continue
            if repo.list_jobs(self.db, statuses=["queued", "running"], project_id=row["project_id"], limit=1):
                continue
            repo.requeue_job(self.db, row["id"], front=True)
            resumed.append(row["id"])
        if resumed:
            log.info("resumed %d interrupted job(s)", len(resumed))
        return list(reversed(resumed))

    # ------------------------------------------------------------------ queue

    def queue_state(self) -> dict[str, Any]:
        with self._lock:
            running = self._running.job_id if self._running else None
        return {
            "paused": bool(repo.get_setting(self.db, SETTING_PAUSED, False)),
            "auto_resume": bool(repo.get_setting(self.db, SETTING_AUTO_RESUME, True)),
            "running": repo.get_job(self.db, running) if running else None,
            "queued": [repo.get_job(self.db, job_id) for job_id in repo.queued_job_ids(self.db)],
        }

    def pause(self, interrupt: bool = False) -> dict[str, Any]:
        """Start no further jobs. With ``interrupt``, also stop the running job
        now; it keeps its progress and is queued again at the front."""
        repo.set_setting(self.db, SETTING_PAUSED, True)
        if interrupt:
            with self._lock:
                running = self._running
                if running is not None and not running.cancel_requested:
                    running.pause_requested = True
                    self._signal(running.process, signal.SIGINT)
        self._queue_changed()
        return self.queue_state()

    def resume_queue(self) -> dict[str, Any]:
        repo.set_setting(self.db, SETTING_PAUSED, False)
        self._queue_changed()
        self._wake.set()
        return self.queue_state()

    def reorder(self, job_ids: list[str]) -> dict[str, Any]:
        repo.reorder_queue(self.db, job_ids)
        for job_id in job_ids:
            self._announce(job_id)
        self._queue_changed()
        return self.queue_state()

    def set_auto_resume(self, enabled: bool) -> dict[str, Any]:
        repo.set_setting(self.db, SETTING_AUTO_RESUME, bool(enabled))
        self._queue_changed()
        return self.queue_state()

    def _queue_changed(self) -> None:
        state = self.queue_state()
        self.notify("queue.updated", {"paused": state["paused"], "auto_resume": state["auto_resume"],
                                      "running": state["running"]["id"] if state["running"] else None,
                                      "queued": [job["id"] for job in state["queued"]]})

    def shutdown(self, timeout: float = CANCEL_GRACE_SECONDS) -> None:
        self._stopping.set()
        self._wake.set()
        with self._lock:
            running = self._running
            if running is not None:
                running.shutdown_requested = True
                self._signal(running.process, signal.SIGINT)  # interrupted, resumable
        if running is not None:
            try:
                running.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                running.process.kill()
                running.process.wait()
            # The monitor records the outcome; wait for it so it never touches
            # the database after the caller has closed it.
            if running.monitor is not None:
                running.monitor.join(timeout=5)
            if running.monitor is None or running.monitor.is_alive():
                self._reconcile(running)
        if self._scheduler is not None:
            self._scheduler.join(timeout=2)

    # ------------------------------------------------------------------ commands

    def submit(self, project_id: str, options: dict[str, Any] | None, kind: str = "transcribe") -> dict[str, Any]:
        """Queue a job. A ``youtube`` job carries the retrieval request in
        ``options["youtube"]`` next to the normalized options."""
        extra = {"youtube": (options or {}).get("youtube")} if kind == "youtube" else {}
        normalized = normalize_options({k: v for k, v in (options or {}).items() if k != "youtube"})
        active = repo.list_jobs(self.db, statuses=["queued", "running"], project_id=project_id, limit=1)
        if active:
            raise InvalidJobState("This project already has a job in progress.", data={"job_id": active[0]["id"]})
        job = repo.create_job(self.db, project_id, normalized["mode"], {**normalized, **extra}, kind=kind)
        self._announce(job["id"])
        self._queue_changed()
        self._wake.set()
        return job

    def submit_recheck(self, project_id: str, segment_ids: list[int]) -> dict[str, Any]:
        """Re-check chosen passages of the current transcript (passes 6-7), with
        the settings and extracted audio of the job that produced it."""
        active = repo.list_jobs(self.db, statuses=["queued", "running"], project_id=project_id, limit=1)
        if active:
            raise InvalidJobState("This project already has a job in progress.", data={"job_id": active[0]["id"]})
        source = next((j for j in repo.list_jobs(self.db, statuses=["completed"], project_id=project_id)
                       if j["kind"] == "transcribe"), None)
        if source is None:
            raise InvalidJobState("Transcribe the recording before re-checking passages.")
        if not (self.paths.job_dir(source["id"]) / "audio.wav").is_file():
            raise InvalidJobState("The audio extracted for this transcript is no longer available. Transcribe the "
                                  "recording again to re-check passages.")
        known = {r["id"] for r in self.db.query("SELECT id FROM segments WHERE project_id = ?", (project_id,))}
        wanted = sorted(set(segment_ids))
        if not wanted or any(sid not in known for sid in wanted):
            raise NotFound("Segment not found in this project.", data={"segment_ids": wanted})
        options = {**source["options"], "recheck": True,
                   "recheck_request": {"segment_ids": wanted, "source_job": source["id"]}}
        job = repo.create_job(self.db, project_id, options["mode"], options, kind="recheck")
        self._announce(job["id"])
        self._queue_changed()
        self._wake.set()
        return job

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = repo.get_job(self.db, job_id)
        if job["status"] == "queued":
            repo.finish_job(self.db, job_id, "cancelled", error_code="cancelled", error_message="Cancelled before it started.")
        elif job["status"] == "running":
            repo.request_cancel(self.db, job_id)
            with self._lock:
                running = self._running
                if running is not None and running.job_id == job_id:
                    running.cancel_requested = True
                    self._signal(running.process, signal.SIGTERM)
                    running.kill_timer = threading.Timer(CANCEL_GRACE_SECONDS, self._force_kill, args=(running,))
                    running.kill_timer.daemon = True
                    running.kill_timer.start()
        else:
            raise InvalidJobState(f"A {job['status']} job cannot be cancelled.", data={"status": job["status"]})
        self._announce(job_id)
        self._queue_changed()
        return repo.get_job(self.db, job_id)

    def resume(self, job_id: str) -> dict[str, Any]:
        job = repo.get_job(self.db, job_id)
        if job["status"] not in repo.RESUMABLE_JOB_STATUSES:
            raise InvalidJobState(f"A {job['status']} job cannot be resumed.", data={"status": job["status"]})
        active = repo.list_jobs(self.db, statuses=["queued", "running"], project_id=job["project_id"], limit=1)
        if active:
            raise InvalidJobState("This project already has a job in progress.", data={"job_id": active[0]["id"]})
        repo.requeue_job(self.db, job_id)
        self._announce(job_id)
        self._queue_changed()
        self._wake.set()
        return repo.get_job(self.db, job_id)

    def cancel_project_jobs(self, project_id: str, wait: float = CANCEL_GRACE_SECONDS + 2) -> None:
        for job in repo.list_jobs(self.db, statuses=["queued", "running"], project_id=project_id):
            self.cancel(job["id"])
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            with self._lock:
                if self._running is None or repo.get_job(self.db, self._running.job_id)["project_id"] != project_id:
                    return
            time.sleep(0.05)

    def running_job_id(self) -> str | None:
        with self._lock:
            return self._running.job_id if self._running else None

    # ------------------------------------------------------------------ internals

    def _announce(self, job_id: str) -> None:
        try:
            job = repo.get_job(self.db, job_id)
        except NotFound:
            return
        self.notify("job.updated", job)
        try:
            self.notify("project.updated", repo.get_project(self.db, job["project_id"]))
        except NotFound:
            pass

    @staticmethod
    def _signal(process: subprocess.Popen, sig: int) -> None:
        try:
            process.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass

    def _force_kill(self, running: _Running) -> None:
        if running.process.poll() is None:
            log.warning("worker for job %s ignored SIGTERM; killing", running.job_id)
            running.process.kill()

    def _schedule_loop(self) -> None:
        while not self._stopping.is_set():
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            if self._stopping.is_set():
                return
            with self._lock:
                if self._running is not None or repo.get_setting(self.db, SETTING_PAUSED, False):
                    continue
                job_id = repo.next_queued_job(self.db)
                if job_id is None:
                    continue
                try:
                    self._spawn(job_id)
                except Exception as exc:  # noqa: BLE001
                    log.exception("could not start worker for job %s", job_id)
                    repo.finish_job(self.db, job_id, "failed", error_code="internal_error",
                                    error_message=f"The transcription worker could not be started: {exc}")
                    self._announce(job_id)

    def _spawn(self, job_id: str) -> None:
        log_path = self.paths.logs_dir / f"worker-{job_id}.log"
        log_file = open(log_path, "ab")  # noqa: SIM115 - closed after the worker exits
        command = [*self.worker_command, "worker", "--job", job_id, "--data-dir", str(self.paths.data_dir)]
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=log_file,
            env=default_env(),
            close_fds=True,
        )
        log_file.close()
        running = _Running(job_id, process, log_path)
        self._running = running
        repo.mark_job_running(self.db, job_id, process.pid)
        self._announce(job_id)
        self._queue_changed()
        running.monitor = threading.Thread(target=self._monitor, args=(running,), name=f"worker-{job_id[:8]}", daemon=True)
        running.monitor.start()

    def _monitor(self, running: _Running) -> None:
        assert running.process.stdout is not None
        for line in running.process.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("event") == "updated":
                self._announce(running.job_id)
        running.process.wait()
        if running.kill_timer is not None:
            running.kill_timer.cancel()
        self._reconcile(running)
        if running.pause_requested and not self._stopping.is_set():
            try:
                if repo.get_job(self.db, running.job_id)["status"] == "interrupted":
                    repo.requeue_job(self.db, running.job_id, front=True)
                    self._announce(running.job_id)
            except NotFound:
                pass
        with self._lock:
            if self._running is running:
                self._running = None
        self._queue_changed()
        self._wake.set()

    def _reconcile(self, running: _Running) -> None:
        """Make the persisted status match what actually happened."""
        try:
            job = repo.get_job(self.db, running.job_id)
        except NotFound:
            return  # project deleted meanwhile
        if job["status"] == "running":
            code = running.process.returncode
            if running.cancel_requested:
                repo.finish_job(self.db, running.job_id, "cancelled", error_code="cancelled", error_message="The job was cancelled.")
            elif running.shutdown_requested or running.pause_requested:
                repo.finish_job(self.db, running.job_id, "interrupted", error_code="interrupted",
                                error_message="Processing stopped because the app quit. Resume to continue."
                                if running.shutdown_requested else "Paused. It continues where it stopped.")
            else:
                how = f"signal {-code}" if code is not None and code < 0 else f"exit code {code}"
                repo.finish_job(
                    self.db, running.job_id, "failed", error_code="worker_crashed",
                    error_message=f"The transcription worker stopped unexpectedly ({how}). "
                    f"Completed chunks are kept; resume to continue. Log: {running.log_path}",
                )
        self._announce(running.job_id)
