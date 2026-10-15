"""Runs explicit, user-initiated model downloads in subprocesses so they can be
cancelled (snapshot_download has no cooperative cancellation)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from typing import Any, Callable

from ..errors import InvalidJobState
from ..paths import EnginePaths
from .catalog import get_spec

log = logging.getLogger(__name__)

Notify = Callable[[str, dict[str, Any]], None]


class DownloadManager:
    def __init__(self, paths: EnginePaths, notify: Notify, command: list[str] | None = None):
        self.paths = paths
        self.notify = notify
        self.command = command or [sys.executable, "-m", "verbatim_engine"]
        self._lock = threading.Lock()
        self._active: dict[str, dict[str, Any]] = {}

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {k: {kk: vv for kk, vv in v.items() if kk != "process"} for k, v in self._active.items()}

    def start(self, model_id: str) -> dict[str, Any]:
        spec = get_spec(model_id)
        with self._lock:
            if model_id in self._active:
                raise InvalidJobState("This model is already downloading.")
            env = dict(os.environ)
            env.pop("HF_HUB_OFFLINE", None)
            env["HF_HUB_DISABLE_TELEMETRY"] = "1"
            env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            log_path = self.paths.logs_dir / f"download-{spec.backend}-{spec.family}.log"
            with open(log_path, "ab") as log_file:
                process = subprocess.Popen(
                    [*self.command, "download-model", model_id, "--data-dir", str(self.paths.data_dir)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    env=env,
                    close_fds=True,
                )
            state = {"model_id": model_id, "status": "running", "done": 0, "total": spec.approx_download_bytes,
                     "total_is_estimate": True, "error": None, "process": process}
            self._active[model_id] = state
        self._emit(state)
        threading.Thread(target=self._monitor, args=(model_id, process), daemon=True).start()
        return {k: v for k, v in state.items() if k != "process"}

    def cancel(self, model_id: str) -> None:
        with self._lock:
            state = self._active.get(model_id)
            if state is None:
                return
            state["status"] = "cancelling"
            try:
                state["process"].terminate()
            except OSError:
                pass

    def shutdown(self) -> None:
        with self._lock:
            processes = [s["process"] for s in self._active.values()]
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()

    def _emit(self, state: dict[str, Any]) -> None:
        self.notify("model.download", {k: v for k, v in state.items() if k != "process"})

    def _monitor(self, model_id: str, process: subprocess.Popen) -> None:
        final: dict[str, Any] | None = None
        assert process.stdout is not None
        for line in process.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            with self._lock:
                state = self._active.get(model_id)
                if state is None:
                    continue
                if event.get("event") == "progress":
                    state["done"], state["total"] = event["done"], event["total"]
                    state["total_is_estimate"] = False
                elif event.get("event") in ("completed", "failed"):
                    final = event
                snapshot = dict(state)
            self._emit(snapshot)
        process.wait()
        with self._lock:
            state = self._active.pop(model_id, None) or {"model_id": model_id}
        if final and final.get("event") == "completed":
            state.update(status="completed", error=None)
        elif state.get("status") == "cancelling":
            state.update(status="cancelled", error=None)
        else:
            message = (final or {}).get("error") or f"The download process exited with code {process.returncode}."
            state.update(status="failed", error=message)
        self._emit(state)


def run_download(model_id: str, data_dir: str | None) -> int:
    """Entry point of the download subprocess."""
    from ..errors import EngineError
    from ..storage.db import Database
    from .manager import ModelManager

    paths = EnginePaths.create(data_dir)
    manager = ModelManager(Database(paths.db_path), paths)

    def emit(event: dict[str, Any]) -> None:
        sys.stdout.write(json.dumps(event) + "\n")
        sys.stdout.flush()

    try:
        status = manager.download(model_id, progress=lambda done, total: emit({"event": "progress", "done": done, "total": total}))
    except EngineError as exc:
        emit({"event": "failed", "error": exc.message, "code": exc.code})
        return 1
    except Exception as exc:  # noqa: BLE001
        emit({"event": "failed", "error": f"{type(exc).__name__}: {exc}"})
        return 1
    emit({"event": "completed", "status": status})
    return 0
