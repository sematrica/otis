"""SQLite connection management and schema migrations.

The server process and the job worker both open the database. WAL mode lets
readers proceed during writes; writers take short ``BEGIN IMMEDIATE``
transactions and wait (busy timeout) for each other.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

SCHEMA_V1 = """
CREATE TABLE projects (
    id               TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    status           TEXT NOT NULL CHECK (status IN
                        ('new','queued','processing','ready','failed','cancelled','interrupted')),
    source_kind      TEXT NOT NULL DEFAULT 'file' CHECK (source_kind IN ('file','url')),
    source_path      TEXT,
    source_url       TEXT,
    source_size      INTEGER,
    source_mtime_ns  INTEGER,
    media_info_json  TEXT NOT NULL,
    duration         REAL,
    has_video        INTEGER NOT NULL DEFAULT 0,
    language         TEXT,
    last_job_id      TEXT
);

CREATE TABLE jobs (
    id              TEXT PRIMARY KEY,
    project_id      TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    kind            TEXT NOT NULL DEFAULT 'transcribe',
    status          TEXT NOT NULL CHECK (status IN
                       ('queued','running','completed','failed','cancelled','interrupted')),
    mode            TEXT NOT NULL,
    options_json    TEXT NOT NULL,
    effective_json  TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT,
    attempt         INTEGER NOT NULL DEFAULT 0,
    worker_pid      INTEGER,
    current_stage   TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    error_code      TEXT,
    error_message   TEXT
);
CREATE INDEX jobs_by_status ON jobs(status, created_at);
CREATE INDEX jobs_by_project ON jobs(project_id, created_at);

CREATE TABLE job_stages (
    job_id          TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    stage           TEXT NOT NULL,
    ordinal         INTEGER NOT NULL,
    status          TEXT NOT NULL CHECK (status IN
                       ('pending','running','completed','failed','skipped','cancelled')),
    started_at      TEXT,
    finished_at     TEXT,
    progress_done   REAL,
    progress_total  REAL,
    progress_unit   TEXT,
    baseline_done   REAL,
    detail          TEXT,
    PRIMARY KEY (job_id, stage)
);

CREATE TABLE audio_assets (
    job_id          TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    path            TEXT NOT NULL,
    sample_rate     INTEGER NOT NULL,
    samples         INTEGER NOT NULL,
    duration        REAL NOT NULL,
    source_size     INTEGER NOT NULL,
    source_mtime_ns INTEGER NOT NULL,
    report_json     TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE chunks (
    job_id               TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    idx                  INTEGER NOT NULL,
    owned_start          REAL NOT NULL,
    owned_end            REAL NOT NULL,
    audio_start          REAL NOT NULL,
    audio_end            REAL NOT NULL,
    cut_energy_db        REAL,
    status               TEXT NOT NULL CHECK (status IN ('pending','completed')),
    result_json          TEXT,
    language             TEXT,
    language_probability REAL,
    processing_seconds   REAL,
    completed_at         TEXT,
    PRIMARY KEY (job_id, idx)
);

-- original_text is what the recogniser produced and is never modified.
-- text is the current text (user corrections arrive in Stage 2).
CREATE TABLE segments (
    id                INTEGER PRIMARY KEY,
    project_id        TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    job_id            TEXT REFERENCES jobs(id) ON DELETE SET NULL,
    ordinal           INTEGER NOT NULL,
    start             REAL NOT NULL,
    end               REAL NOT NULL,
    original_text     TEXT NOT NULL,
    text              TEXT NOT NULL,
    language          TEXT,
    avg_logprob       REAL,
    no_speech_prob    REAL,
    compression_ratio REAL,
    temperature       REAL,
    chunk_index       INTEGER,
    speaker_id        INTEGER,
    flags_json        TEXT NOT NULL DEFAULT '[]',
    UNIQUE (project_id, ordinal)
);
CREATE INDEX segments_by_time ON segments(project_id, start);

CREATE TABLE words (
    id            INTEGER PRIMARY KEY,
    segment_id    INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    ordinal       INTEGER NOT NULL,
    start         REAL,
    end           REAL,
    text          TEXT NOT NULL,
    probability   REAL,
    timing_source TEXT NOT NULL DEFAULT 'whisper',
    UNIQUE (segment_id, ordinal)
);

CREATE TABLE settings (
    key        TEXT PRIMARY KEY,
    value_json TEXT NOT NULL
);
"""

MIGRATIONS: list[tuple[int, str]] = [(1, SCHEMA_V1)]
LATEST_VERSION = MIGRATIONS[-1][0]


class Database:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self.migrate()

    def migrate(self) -> None:
        with self._lock:
            current = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if current > LATEST_VERSION:
                raise RuntimeError(
                    f"Database schema version {current} is newer than this engine supports ({LATEST_VERSION})."
                )
            for version, sql in MIGRATIONS:
                if version <= current:
                    continue
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    # re-check inside the write lock: another process may have migrated
                    if self._conn.execute("PRAGMA user_version").fetchone()[0] >= version:
                        self._conn.execute("COMMIT")
                        continue
                    for statement in _split_sql(sql):
                        self._conn.execute(statement)
                    self._conn.execute(f"PRAGMA user_version = {int(version)}")
                    self._conn.execute("COMMIT")
                except BaseException:
                    self._conn.execute("ROLLBACK")
                    raise

    @property
    def schema_version(self) -> int:
        with self._lock:
            return self._conn.execute("PRAGMA user_version").fetchone()[0]

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def execute(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def query(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _split_sql(script: str) -> Iterable[str]:
    """Split a schema script into statements (the schema contains no
    semicolons inside literals or triggers)."""
    lines = [line for line in script.splitlines() if not line.strip().startswith("--")]
    for statement in "\n".join(lines).split(";"):
        if statement.strip():
            yield statement.strip()
