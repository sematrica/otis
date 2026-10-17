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

from ..errors import EngineError

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
-- text is the current text (including user corrections).
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

# Stage 2: speakers, per-word speaker and alignment data, an immutable copy of
# the AI-generated transcript, and an edit history for undo/redo.
# The segments table is rebuilt (SQLite cannot drop a UNIQUE constraint):
# segment order is now by time, so splitting or merging segments never has
# to renumber the rest of the transcript.
SCHEMA_V2 = """
CREATE TABLE speakers (
    id            INTEGER PRIMARY KEY,
    project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    label         TEXT NOT NULL,
    name          TEXT,
    origin        TEXT NOT NULL CHECK (origin IN ('diarization','user')),
    cluster_index INTEGER,
    color_index   INTEGER NOT NULL DEFAULT 0,
    merged_into   INTEGER REFERENCES speakers(id) ON DELETE SET NULL,
    created_at    TEXT NOT NULL
);
CREATE INDEX speakers_by_project ON speakers(project_id);

CREATE TABLE segments_v2 (
    id                  INTEGER PRIMARY KEY,
    project_id          TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    job_id              TEXT REFERENCES jobs(id) ON DELETE SET NULL,
    ordinal             INTEGER NOT NULL,
    start               REAL NOT NULL,
    end                 REAL NOT NULL,
    original_text       TEXT NOT NULL,
    text                TEXT NOT NULL,
    language            TEXT,
    avg_logprob         REAL,
    no_speech_prob      REAL,
    compression_ratio   REAL,
    temperature         REAL,
    chunk_index         INTEGER,
    speaker_id          INTEGER REFERENCES speakers(id) ON DELETE SET NULL,
    original_speaker_id INTEGER REFERENCES speakers(id) ON DELETE SET NULL,
    speaker_confidence  REAL,
    text_edited         INTEGER NOT NULL DEFAULT 0,
    speaker_edited      INTEGER NOT NULL DEFAULT 0,
    flags_json          TEXT NOT NULL DEFAULT '[]'
);
INSERT INTO segments_v2 (id, project_id, job_id, ordinal, start, end, original_text, text, language, avg_logprob,
                         no_speech_prob, compression_ratio, temperature, chunk_index, flags_json)
    SELECT id, project_id, job_id, ordinal, start, end, original_text, text, language, avg_logprob,
           no_speech_prob, compression_ratio, temperature, chunk_index, flags_json
    FROM segments;
DROP TABLE segments;
ALTER TABLE segments_v2 RENAME TO segments;
CREATE INDEX segments_by_time ON segments(project_id, start);

ALTER TABLE words ADD COLUMN speaker_id INTEGER REFERENCES speakers(id) ON DELETE SET NULL;
ALTER TABLE words ADD COLUMN whisper_start REAL;
ALTER TABLE words ADD COLUMN whisper_end REAL;
ALTER TABLE words ADD COLUMN alignment_score REAL;
ALTER TABLE words ADD COLUMN original_text TEXT;
UPDATE words SET original_text = text;

CREATE TABLE original_transcripts (
    project_id   TEXT PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
    job_id       TEXT,
    created_at   TEXT NOT NULL,
    content_json TEXT NOT NULL
);

CREATE TABLE edits (
    id          INTEGER PRIMARY KEY,
    project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,
    summary     TEXT NOT NULL,
    before_json TEXT NOT NULL,
    after_json  TEXT NOT NULL,
    undone      INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);
CREATE INDEX edits_by_project ON edits(project_id, id);

ALTER TABLE projects ADD COLUMN transcript_revision INTEGER NOT NULL DEFAULT 0;
"""

SCHEMA_V3 = """
-- Stage 3: AI verification runs and their suggestions. (jobs.kind exists since
-- version 1; Stage 3 adds the 'recheck' kind.)

CREATE TABLE llm_runs (
    id                  INTEGER PRIMARY KEY,
    project_id          TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    provider            TEXT NOT NULL,
    model               TEXT NOT NULL,
    endpoint            TEXT NOT NULL,
    cloud               INTEGER NOT NULL,
    status              TEXT NOT NULL CHECK (status IN ('running','completed','failed','cancelled')),
    error               TEXT,
    transcript_revision INTEGER NOT NULL,
    segments_sent       INTEGER NOT NULL DEFAULT 0,
    requests            INTEGER NOT NULL DEFAULT 0,
    exchanges_json      TEXT NOT NULL DEFAULT '[]',
    started_at          TEXT NOT NULL,
    finished_at         TEXT
);
CREATE INDEX llm_runs_by_project ON llm_runs(project_id, id);

CREATE TABLE suggestions (
    id             INTEGER PRIMARY KEY,
    project_id     TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    run_id         INTEGER NOT NULL REFERENCES llm_runs(id) ON DELETE CASCADE,
    segment_id     INTEGER REFERENCES segments(id) ON DELETE SET NULL,
    kind           TEXT NOT NULL CHECK (kind IN ('suspicious','punctuation','terminology','retranscribe')),
    original_text  TEXT NOT NULL,
    suggested_text TEXT,
    reason         TEXT,
    status         TEXT NOT NULL DEFAULT 'pending'
                   CHECK (status IN ('pending','accepted','rejected','invalid','outdated')),
    validation     TEXT,
    created_at     TEXT NOT NULL,
    decided_at     TEXT
);
CREATE INDEX suggestions_by_project ON suggestions(project_id, status);
"""

#: (version, script, rebuilds_tables). Table rebuilds run with foreign-key
#: enforcement off, as SQLite requires, and are checked afterwards.
SCHEMA_V4 = """
-- Stage 4: full-text search over transcripts. An external-content FTS5 index
-- of the segments' current and AI text, kept in sync by triggers (every
-- correction, undo, re-check and new transcription goes through them).
-- unicode61 with diacritics removed: "cafe" finds "café".

CREATE VIRTUAL TABLE segments_fts USING fts5(
    text, original_text,
    content='segments', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER segments_fts_insert AFTER INSERT ON segments BEGIN
    INSERT INTO segments_fts(rowid, text, original_text) VALUES (new.id, new.text, new.original_text);
END;

CREATE TRIGGER segments_fts_delete AFTER DELETE ON segments BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, text, original_text)
        VALUES ('delete', old.id, old.text, old.original_text);
END;

CREATE TRIGGER segments_fts_update AFTER UPDATE OF text, original_text ON segments BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, text, original_text)
        VALUES ('delete', old.id, old.text, old.original_text);
    INSERT INTO segments_fts(rowid, text, original_text) VALUES (new.id, new.text, new.original_text);
END;

INSERT INTO segments_fts(segments_fts) VALUES ('rebuild');
"""

SCHEMA_V5 = """
-- Stage 4: a persistent, ordered processing queue. Queued jobs run in
-- queue_position order (the order they were added until the user reorders
-- them); existing jobs keep their insertion order.
ALTER TABLE jobs ADD COLUMN queue_position REAL;
UPDATE jobs SET queue_position = rowid;
CREATE INDEX jobs_queue ON jobs(status, queue_position);
"""

SCHEMA_V6 = """
-- Stage 4: media retrieved from YouTube. source_meta_json describes the
-- source (video id, title, channel, licence, chapters, the user's
-- confirmation that they may retrieve it, what was retrieved); captions
-- published with it are kept separately and never merged into a transcript.
ALTER TABLE projects ADD COLUMN source_meta_json TEXT;

CREATE TABLE captions (
    id          INTEGER PRIMARY KEY,
    project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    language    TEXT NOT NULL,
    name        TEXT,
    kind        TEXT NOT NULL CHECK (kind IN ('manual','auto')),
    source      TEXT NOT NULL,
    fetched_at  TEXT NOT NULL,
    cues_json   TEXT NOT NULL
);
CREATE INDEX captions_by_project ON captions(project_id);
"""

MIGRATIONS: list[tuple[int, str, bool]] = [(1, SCHEMA_V1, False), (2, SCHEMA_V2, True), (3, SCHEMA_V3, False),
                                           (4, SCHEMA_V4, False), (5, SCHEMA_V5, False), (6, SCHEMA_V6, False)]
LATEST_VERSION = MIGRATIONS[-1][0]


class DatabaseTooNew(EngineError):
    code = "database_too_new"


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
                raise DatabaseTooNew(
                    f"The library in {Path(self.path).parent} was last opened by a newer version of Verbatim "
                    f"(database version {current}; this version reads up to {LATEST_VERSION}). Open it with "
                    "the newer version.",
                    data={"schema_version": current, "supported": LATEST_VERSION})
            for version, sql, rebuilds in MIGRATIONS:
                if version <= current:
                    continue
                if rebuilds:
                    # must be set outside a transaction; otherwise dropping the
                    # old table would cascade-delete rows that reference it
                    self._conn.execute("PRAGMA foreign_keys=OFF")
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    # re-check inside the write lock: another process may have migrated
                    if self._conn.execute("PRAGMA user_version").fetchone()[0] >= version:
                        self._conn.execute("COMMIT")
                        continue
                    for statement in _split_sql(sql):
                        self._conn.execute(statement)
                    if rebuilds:
                        problems = self._conn.execute("PRAGMA foreign_key_check").fetchall()
                        if problems:
                            raise RuntimeError(f"Migration {version} left broken references: {problems[:5]}")
                    self._conn.execute(f"PRAGMA user_version = {int(version)}")
                    self._conn.execute("COMMIT")
                except BaseException:
                    self._conn.execute("ROLLBACK")
                    raise
                finally:
                    if rebuilds:
                        self._conn.execute("PRAGMA foreign_keys=ON")

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

    @contextmanager
    def snapshot(self) -> Iterator[sqlite3.Connection]:
        """A read transaction: consistent view, no write lock."""
        with self._lock:
            self._conn.execute("BEGIN DEFERRED")
            try:
                yield self._conn
            finally:
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
    """Split a schema script into complete statements (trigger bodies keep
    their inner semicolons; the schema has no semicolons inside literals)."""
    lines = [line for line in script.splitlines() if not line.strip().startswith("--")]
    buffer = ""
    for piece in "\n".join(lines).split(";"):
        buffer += piece + ";"
        if sqlite3.complete_statement(buffer):
            if buffer.strip(" \n\t;"):
                yield buffer.strip()
            buffer = ""
    if buffer.strip(" \n\t;"):
        raise ValueError(f"Incomplete SQL statement in schema: {buffer.strip()[:80]}")
