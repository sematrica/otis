"""JSON-lines RPC over stdin/stdout.

* request:      {"id": <int|str>, "method": "<name>", "params": {...}}
* success:      {"id": ..., "result": ...}
* failure:      {"id": ..., "error": {"code": "<engine code>", "message": "...", "data": {...}}}
* notification: {"method": "<name>", "params": {...}}   (engine → app, no id)

stdout carries only protocol messages. At start-up the real stdout file
descriptor is duplicated for the protocol and fd 1 is pointed at stderr, so
stray prints from any library can never corrupt the stream.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import IO, Any

from ..errors import EngineError
from ..paths import EnginePaths
from .service import EngineService

log = logging.getLogger(__name__)


class ProtocolWriter:
    def __init__(self, stream: IO[str]):
        self._stream = stream
        self._lock = threading.Lock()
        self.closed = False

    def send(self, message: dict[str, Any]) -> None:
        line = json.dumps(message, ensure_ascii=True, separators=(",", ":"), default=str)
        with self._lock:
            if self.closed:
                return
            try:
                self._stream.write(line + "\n")
                self._stream.flush()
            except (BrokenPipeError, ValueError, OSError):
                self.closed = True

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self.send({"method": method, "params": params})


#: The engine's last stderr line when it cannot start; the app shows what follows.
STARTUP_ERROR_PREFIX = "verbatim-engine: cannot start: "
STARTUP_FAILED = 3


def _claim_stdout() -> IO[str]:
    protocol_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    return os.fdopen(protocol_fd, "w", encoding="utf-8", newline="\n", buffering=1)


def handle_line(service: EngineService, methods, writer: ProtocolWriter, line: str) -> None:
    try:
        message = json.loads(line)
    except ValueError:
        writer.send({"id": None, "error": {"code": "parse_error", "message": "Invalid JSON."}})
        return
    if not isinstance(message, dict) or not isinstance(message.get("method"), str):
        writer.send({"id": message.get("id") if isinstance(message, dict) else None,
                     "error": {"code": "invalid_request", "message": "Expected an object with a 'method'."}})
        return
    request_id = message.get("id")
    method = message["method"]
    params = message.get("params") or {}
    handler = methods.get(method)
    if handler is None:
        writer.send({"id": request_id, "error": {"code": "method_not_found", "message": f"Unknown method '{method}'."}})
        return
    if not isinstance(params, dict):
        writer.send({"id": request_id, "error": {"code": "invalid_params", "message": "'params' must be an object."}})
        return
    try:
        result = handler(params)
    except EngineError as exc:
        writer.send({"id": request_id, "error": exc.to_dict()})
    except Exception as exc:  # noqa: BLE001 - never let one request kill the server
        log.exception("method %s failed", method)
        writer.send({"id": request_id, "error": {"code": "internal_error", "message": f"{type(exc).__name__}: {exc}"}})
    else:
        writer.send({"id": request_id, "result": result})


def _method_of(line: str) -> str | None:
    try:
        message = json.loads(line)
    except ValueError:
        return None
    return message.get("method") if isinstance(message, dict) else None


def serve(data_dir: str | None = None, worker_command: list[str] | None = None) -> int:
    protocol_out = _claim_stdout()
    writer = ProtocolWriter(protocol_out)
    try:
        paths = EnginePaths.create(data_dir)
        service = EngineService(paths, writer.notify, worker_command)
        service.start()
    except Exception as exc:  # noqa: BLE001 - reported to the app, which shows it
        log.exception("the engine could not start")
        reason = exc.message if isinstance(exc, EngineError) else f"{type(exc).__name__}: {exc}"
        print(STARTUP_ERROR_PREFIX + " ".join(reason.split()), file=sys.stderr, flush=True)
        return STARTUP_FAILED
    methods = service.methods()
    log.info("engine serving (data dir %s)", paths.data_dir)
    # Requests run concurrently so a slow call (e.g. deleting a project with a
    # running job) never blocks quick ones. Responses carry the request id.
    pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rpc")
    try:
        for raw in sys.stdin.buffer:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if _method_of(line) == "engine.shutdown":
                handle_line(service, methods, writer, line)
                break
            pool.submit(handle_line, service, methods, writer, line)
    finally:
        pool.shutdown(wait=True)
        service.shutdown()
        writer.closed = True
        log.info("engine stopped")
    return 0
