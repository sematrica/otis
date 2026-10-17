"""AI review runs: storage and the background runner.

A run sends a project's transcript to one provider in batches, validates and
stores the suggestions, and keeps every request and response (``exchanges``)
so each suggestion can be traced to what the model was asked and answered.
Runs execute in a background thread of the engine server; the app is told
about progress with ``llm.run.updated`` notifications.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable

from ..errors import EngineError, InvalidParams, NotFound
from ..storage import repo
from ..storage.db import Database
from ..util.ids import utc_now
from .providers import LLMError, Provider
from .review import SCHEMA, SYSTEM_PROMPT, batches, collect, prompt_for

log = logging.getLogger(__name__)

Notify = Callable[[str, dict[str, Any]], None]


# ----------------------------------------------------------------------------- storage


def _run_dict(row, exchanges: bool = False) -> dict[str, Any]:
    out = {k: row[k] for k in ("id", "project_id", "provider", "model", "endpoint", "status", "error",
                               "transcript_revision", "segments_sent", "requests", "started_at", "finished_at")}
    out["remote"] = bool(row["cloud"])
    if exchanges:
        out["exchanges"] = json.loads(row["exchanges_json"])
    return out


def get_run(db: Database, run_id: int, exchanges: bool = False) -> dict[str, Any]:
    row = db.query_one("SELECT * FROM llm_runs WHERE id = ?", (run_id,))
    if row is None:
        raise NotFound("Review run not found.", data={"run_id": run_id})
    out = _run_dict(row, exchanges)
    out["suggestions"] = {
        r["status"]: r["n"] for r in db.query(
            "SELECT status, COUNT(*) AS n FROM suggestions WHERE run_id = ? GROUP BY status", (run_id,))
    }
    return out


def list_runs(db: Database, project_id: str) -> list[dict[str, Any]]:
    rows = db.query("SELECT id FROM llm_runs WHERE project_id = ? ORDER BY id DESC LIMIT 50", (project_id,))
    return [get_run(db, r["id"]) for r in rows]


def _suggestion_dict(row) -> dict[str, Any]:
    return {k: row[k] for k in ("id", "project_id", "run_id", "segment_id", "kind", "original_text",
                                "suggested_text", "reason", "status", "validation", "created_at", "decided_at")}


def list_suggestions(db: Database, project_id: str, statuses: list[str] | None = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM suggestions WHERE project_id = ?"
    params: list[Any] = [project_id]
    if statuses:
        sql += f" AND status IN ({','.join('?' * len(statuses))})"
        params.extend(statuses)
    sql += " ORDER BY id"
    return [_suggestion_dict(r) for r in db.query(sql, params)]


def get_suggestion(db: Database, suggestion_id: int) -> dict[str, Any]:
    row = db.query_one("SELECT * FROM suggestions WHERE id = ?", (suggestion_id,))
    if row is None:
        raise NotFound("Suggestion not found.", data={"suggestion_id": suggestion_id})
    return _suggestion_dict(row)


def decide(db: Database, suggestion_id: int, status: str) -> dict[str, Any]:
    if status not in ("accepted", "rejected", "outdated"):
        raise InvalidParams(f"Unknown decision '{status}'.")
    db.execute("UPDATE suggestions SET status = ?, decided_at = ? WHERE id = ?", (status, utc_now(), suggestion_id))
    return get_suggestion(db, suggestion_id)


# ----------------------------------------------------------------------------- runner


class ReviewManager:
    def __init__(self, db: Database, notify: Notify):
        self.db = db
        self.notify = notify
        self._lock = threading.Lock()
        self._threads: dict[int, threading.Thread] = {}
        self._cancel: set[int] = set()

    def start(self, project_id: str, provider: Provider, *, allow_remote: bool = False) -> dict[str, Any]:
        if provider.remote and not allow_remote:
            raise InvalidParams(
                f"{provider.display_name} at {provider.base_url} is not on this computer. The transcript text is only "
                "sent there after you confirm it.", data={"remote": True})
        transcript = repo.get_transcript(self.db, project_id, include_words=False)
        segments = [s for s in transcript["segments"] if s["text"].strip()]
        if not segments:
            raise NotFound("This project has no transcript to review.")
        with self._lock:
            if any(t.is_alive() for t in self._threads.values()) and self._running_for(project_id):
                raise EngineError("An AI review of this project is already running.", code="invalid_state")
        speakers = {sp["id"]: sp.get("display_name") or sp["label"] for sp in transcript.get("speakers", [])}
        cur = self.db.execute(
            """INSERT INTO llm_runs (project_id, provider, model, endpoint, cloud, status, transcript_revision,
                                     started_at) VALUES (?, ?, ?, ?, ?, 'running', ?, ?)""",
            (project_id, provider.name, provider.model, provider.base_url, int(provider.remote),
             transcript["revision"], utc_now()),
        )
        run_id = int(cur.lastrowid)
        thread = threading.Thread(target=self._run, args=(run_id, project_id, provider, segments, speakers),
                                  name=f"llm-review-{run_id}", daemon=True)
        with self._lock:
            self._threads[run_id] = thread
        thread.start()
        return get_run(self.db, run_id)

    def _running_for(self, project_id: str) -> bool:
        return self.db.query_one("SELECT 1 FROM llm_runs WHERE project_id = ? AND status = 'running'",
                                 (project_id,)) is not None

    def cancel(self, run_id: int) -> dict[str, Any]:
        with self._lock:
            self._cancel.add(run_id)
        return get_run(self.db, run_id)

    def shutdown(self, timeout: float = 5.0) -> None:
        with self._lock:
            threads = list(self._threads.items())
            self._cancel.update(run_id for run_id, _ in threads)
        for _, thread in threads:
            thread.join(timeout=timeout)

    def _announce(self, run_id: int) -> None:
        try:
            self.notify("llm.run.updated", get_run(self.db, run_id))
        except Exception:  # noqa: BLE001 - notifications are best effort
            log.debug("could not announce review run %s", run_id, exc_info=True)

    def _record(self, run_id: int, exchange: dict[str, Any], sent: int) -> None:
        with self.db.transaction() as conn:
            row = conn.execute("SELECT exchanges_json FROM llm_runs WHERE id = ?", (run_id,)).fetchone()
            exchanges = json.loads(row["exchanges_json"]) + [exchange]
            conn.execute(
                "UPDATE llm_runs SET exchanges_json = ?, requests = requests + 1, segments_sent = segments_sent + ? "
                "WHERE id = ?", (json.dumps(exchanges), sent, run_id))

    def _run(self, run_id: int, project_id: str, provider: Provider, segments: list[dict[str, Any]],
             speakers: dict[int, str]) -> None:
        status, error = "completed", None
        try:
            for batch in batches(segments, speakers):
                with self._lock:
                    if run_id in self._cancel:
                        status = "cancelled"
                        break
                try:
                    parsed, exchange = provider.complete_json(SYSTEM_PROMPT, prompt_for(batch, speakers), SCHEMA)
                except LLMError as exc:
                    if "exchange" in exc.data:
                        self._record(run_id, exc.data["exchange"], len(batch))
                    raise
                self._record(run_id, exchange.to_dict(), len(batch))
                records = collect(parsed, batch)
                now = utc_now()
                with self.db.transaction() as conn:
                    for r in records:
                        conn.execute(
                            """INSERT INTO suggestions (project_id, run_id, segment_id, kind, original_text,
                                   suggested_text, reason, status, validation, created_at)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (project_id, run_id, r["segment_id"], r["kind"], r["original_text"], r["suggested_text"],
                             r["reason"], r["status"], r["validation"], now),
                        )
                self._announce(run_id)
        except EngineError as exc:
            status, error = "failed", exc.message
        except Exception as exc:  # noqa: BLE001 - keep the server alive; record what happened
            log.exception("AI review run %s failed", run_id)
            status, error = "failed", f"Unexpected error: {type(exc).__name__}: {exc}"
        finally:
            self.db.execute("UPDATE llm_runs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                            (status, error, utc_now(), run_id))
            with self._lock:
                self._threads.pop(run_id, None)
                self._cancel.discard(run_id)
            self._announce(run_id)

    def wait(self, run_id: int, timeout: float | None = None) -> dict[str, Any]:
        """Block until a run finishes (used by tests and the CLI)."""
        with self._lock:
            thread = self._threads.get(run_id)
        if thread is not None:
            thread.join(timeout)
        return get_run(self.db, run_id)
