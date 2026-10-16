"""Persistent Job Manager (server side).

Jobs are queued in the database and run one at a time — on a 16 GB machine
two large models must never be loaded together. Each job runs in a worker
subprocess; this manager spawns it, relays its progress, enforces
cancellation, and reconciles the job's final state when the worker exits for
any reason (including native crashes).
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


class _Running:
    def __init__(self, job_id: str, process: subprocess.Popen, log_path: Path):
        self.job_id = job_id
        self.process = process
        self.log_path = log_path
        self.cancel_requested = False
        self.shutdown_requested = False
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

    def submit(self, project_id: str, options: dict[str, Any] | None) -> dict[str, Any]:
        normalized = normalize_options(options)
        active = repo.list_jobs(self.db, statuses=["queued", "running"], project_id=project_id, limit=1)
        if active:
            raise InvalidJobState("This project already has a job in progress.", data={"job_id": active[0]["id"]})
        job = repo.create_job(self.db, project_id, normalized["mode"], normalized)
        self._announce(job["id"])
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
                if self._running is not None:
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
        with self._lock:
            if self._running is running:
                self._running = None
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
            elif running.shutdown_requested:
                repo.finish_job(self.db, running.job_id, "interrupted", error_code="interrupted",
                                error_message="Processing stopped because the app quit. Resume to continue.")
            else:
                how = f"signal {-code}" if code is not None and code < 0 else f"exit code {code}"
                repo.finish_job(
                    self.db, running.job_id, "failed", error_code="worker_crashed",
                    error_message=f"The transcription worker stopped unexpectedly ({how}). "
                    f"Completed chunks are kept; resume to continue. Log: {running.log_path}",
                )
        self._announce(running.job_id)
