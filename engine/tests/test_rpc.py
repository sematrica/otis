"""The JSON-lines RPC server, exercised as a real subprocess — exactly how the
macOS app talks to it."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import pytest


class EngineProcess:
    def __init__(self, data_dir: Path):
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "verbatim_engine", "serve", "--data-dir", str(data_dir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.lines: Queue = Queue()
        self.raw_stdout: list[str] = []
        self.notifications: list[dict] = []
        self._next_id = 0
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=lambda: self.proc.stderr.read(), daemon=True).start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            self.raw_stdout.append(line)
            self.lines.put(json.loads(line))

    def call(self, method: str, params: dict | None = None, timeout: float = 30.0) -> dict:
        self._next_id += 1
        request_id = self._next_id
        self.proc.stdin.write(json.dumps({"id": request_id, "method": method, "params": params or {}}) + "\n")
        self.proc.stdin.flush()
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(method)
            try:
                message = self.lines.get(timeout=remaining)
            except Empty as exc:
                raise TimeoutError(method) from exc
            if "id" in message and message["id"] == request_id:
                return message
            if "id" not in message:
                self.notifications.append(message)

    def result(self, method: str, params: dict | None = None, timeout: float = 30.0):
        message = self.call(method, params, timeout)
        assert "error" not in message, message
        return message["result"]

    def wait_for_notification(self, predicate, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        for n in self.notifications:
            if predicate(n):
                return n
        while time.monotonic() < deadline:
            try:
                message = self.lines.get(timeout=0.2)
            except Empty:
                continue
            if "id" not in message:
                self.notifications.append(message)
                if predicate(message):
                    return message
        raise TimeoutError("notification")

    def close(self) -> int:
        if self.proc.poll() is None:
            try:
                self.call("engine.shutdown", timeout=30)
            except (TimeoutError, BrokenPipeError):
                pass
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        return self.proc.returncode


@pytest.fixture()
def engine(data_dir):
    process = EngineProcess(data_dir)
    yield process
    process.close()


def test_info_and_protocol_hygiene(engine):
    info = engine.result("engine.info")
    assert info["protocol_version"] == 3
    assert {b["name"] for b in info["backends"]} == {"faster-whisper", "mlx"}
    fw = next(b for b in info["backends"] if b["name"] == "faster-whisper")
    assert fw["available"] is True and fw["capabilities"]["beam_search"] is True
    mlx = next(b for b in info["backends"] if b["name"] == "mlx")
    assert mlx["capabilities"]["beam_search"] is False
    modes = engine.result("transcription.modes")["modes"]
    assert [m["id"] for m in modes] == ["fast", "balanced", "maximum"]
    # every stdout line so far is a protocol message
    assert all(json.loads(line) for line in engine.raw_stdout)


def test_errors_are_structured(engine):
    assert engine.call("no.such.method")["error"]["code"] == "method_not_found"
    assert engine.call("projects.get", {})["error"]["code"] == "invalid_params"
    assert engine.call("projects.get", {"project_id": "nope"})["error"]["code"] == "not_found"
    assert engine.call("media.probe", {"path": "/definitely/missing.mp4"})["error"]["code"] == "media_invalid"
    error = engine.call("transcription.preflight", {"options": {"mode": "maximum", "backend": "faster-whisper"}})["error"]
    assert error["code"] == "model_not_installed"
    assert error["data"]["model_id"] == "faster-whisper/large-v3"
    # malformed input does not kill the server
    engine.proc.stdin.write("{not json\n")
    engine.proc.stdin.flush()
    assert engine.result("engine.info")["protocol_version"] == 3


def test_project_crud_and_export_guard(engine, jfk_path, tmp_path):
    project = engine.result("projects.create", {"path": str(jfk_path)})
    assert project["title"] == "jfk" and project["status"] == "new"
    assert project["media"]["audio_streams"][0]["sample_rate"] == 16000
    listed = engine.result("projects.list")["projects"]
    assert [p["id"] for p in listed] == [project["id"]]
    renamed = engine.result("projects.rename", {"project_id": project["id"], "title": "Inaugural"})
    assert renamed["title"] == "Inaugural"
    assert engine.call("export.render", {"project_id": project["id"], "format": "txt"})["error"]["code"] == "not_found"
    assert engine.call("export.render", {"project_id": project["id"], "format": "docx"})["error"]["code"] in ("invalid_params", "not_found")
    # jobs refuse to queue without an installed model
    error = engine.call("jobs.start", {"project_id": project["id"], "options": {"backend": "faster-whisper"}})["error"]
    assert error["code"] == "model_not_installed"
    assert engine.result("jobs.list", {"active_only": True})["jobs"] == []
    engine.result("projects.delete", {"project_id": project["id"]})
    assert engine.result("projects.list")["projects"] == []
    assert any(n["method"] == "project.deleted" for n in engine.notifications)


def test_models_and_settings(engine, tmp_path):
    listing = engine.result("models.list")
    kinds = [m["kind"] for m in listing["models"]]
    assert kinds.count("asr") == 8 and kinds.count("diarization") == 1 and kinds.count("alignment") == 3
    assert all(m["status"]["installed"] is False for m in listing["models"])
    assert all(m["license"] for m in listing["models"] if m["kind"] != "asr")
    external = tmp_path / "ssd" / "models"
    settings = engine.result("settings.set", {"models_dir": str(external)})
    assert settings["models_dir"] == str(external.resolve())
    assert engine.result("models.list")["models_dir"] == str(external.resolve())
    assert engine.call("settings.set", {"models_dir": "relative"})["error"]["code"] == "invalid_params"


def test_shutdown_exits_cleanly(data_dir):
    process = EngineProcess(data_dir)
    process.result("engine.info")
    assert process.close() == 0
