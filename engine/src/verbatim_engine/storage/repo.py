"""Data access for projects, jobs, stages, chunks and transcripts.

Functions return plain dicts shaped for the RPC protocol
(see docs/ENGINE_PROTOCOL.md).
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from ..audio.chunking import Chunk
from ..errors import InvalidJobState, NotFound
from ..media.probe import MediaInfo
from ..transcript.model import TSegment
from ..verify.review import review_priority
from ..util.ids import new_id, parse_utc, utc_now
from .db import Database

PROJECT_STATUSES = ("new", "queued", "processing", "ready", "failed", "cancelled", "interrupted")
ACTIVE_JOB_STATUSES = ("queued", "running")
RESUMABLE_JOB_STATUSES = ("failed", "cancelled", "interrupted")

#: Pipeline stages in execution order with user-facing labels.
STAGES: list[tuple[str, str]] = [
    ("download", "Retrieving from YouTube"),
    ("prepare", "Checking source"),
    ("extract", "Extracting audio"),
    ("analyze", "Analyzing audio"),
    ("load_model", "Loading model"),
    ("transcribe", "Transcribing"),
    ("diarize", "Identifying speakers"),
    ("align", "Aligning words"),
    ("quality", "Checking quality"),
    ("retranscribe", "Re-checking doubtful passages"),
    ("validate", "Validating transcript"),
    ("finalize", "Saving transcript"),
]
STAGE_ORDINALS = {name: i for i, (name, _label) in enumerate(STAGES)}
#: Stages each kind of job goes through.
JOB_KIND_STAGES = {
    "transcribe": [name for name, _label in STAGES if name != "download"],
    "recheck": ["prepare", "retranscribe", "finalize"],
    # a URL project's first job: retrieve the media, then transcribe it
    "youtube": [name for name, _label in STAGES],
}
STAGE_LABELS = dict(STAGES)

#: ETA is shown only after this much measured work in the current attempt.
_ETA_MIN_FRACTION = 0.03
_ETA_MIN_SECONDS = 10.0


# --------------------------------------------------------------------------- projects


def _project_row_to_dict(row, segment_count: int | None = None) -> dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "source": {
            "kind": row["source_kind"],
            "path": row["source_path"],
            "url": row["source_url"],
            "size_bytes": row["source_size"],
            "mtime_ns": row["source_mtime_ns"],
        },
        "duration": row["duration"],
        "has_video": bool(row["has_video"]),
        "language": row["language"],
        # None until a URL project's media has been retrieved
        "media": json.loads(row["media_info_json"]) or None,
        "last_job_id": row["last_job_id"],
        "segment_count": segment_count if segment_count is not None else 0,
        "source_meta": json.loads(row["source_meta_json"]) if row["source_meta_json"] else None,
    }


def create_project(db: Database, media: MediaInfo, title: str) -> dict[str, Any]:
    project_id = new_id()
    now = utc_now()
    with db.transaction() as conn:
        conn.execute(
            """INSERT INTO projects (id, title, created_at, updated_at, status, source_kind, source_path,
                   source_size, source_mtime_ns, media_info_json, duration, has_video)
               VALUES (?, ?, ?, ?, 'new', 'file', ?, ?, ?, ?, ?, ?)""",
            (
                project_id,
                title,
                now,
                now,
                media.path,
                media.size_bytes,
                media.mtime_ns,
                json.dumps(media.to_dict()),
                media.duration,
                int(media.has_video),
            ),
        )
    return get_project(db, project_id)


def get_project(db: Database, project_id: str) -> dict[str, Any]:
    row = db.query_one("SELECT * FROM projects WHERE id = ?", (project_id,))
    if row is None:
        raise NotFound("Project not found.", data={"project_id": project_id})
    count = db.query_one("SELECT COUNT(*) FROM segments WHERE project_id = ?", (project_id,))[0]
    return _project_row_to_dict(row, count)


def list_projects(db: Database, status: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    sql = (
        "SELECT p.*, (SELECT COUNT(*) FROM segments s WHERE s.project_id = p.id) AS segment_count "
        "FROM projects p"
    )
    params: list[Any] = []
    if status:
        sql += " WHERE p.status = ?"
        params.append(status)
    sql += " ORDER BY p.updated_at DESC"
    if limit:
        sql += " LIMIT ?"
        params.append(int(limit))
    return [_project_row_to_dict(r, r["segment_count"]) for r in db.query(sql, params)]


def create_url_project(db: Database, url: str, title: str, meta: dict[str, Any], *, has_video: bool) -> dict[str, Any]:
    """A project whose media is retrieved from ``url`` by its first job."""
    project_id = new_id()
    now = utc_now()
    with db.transaction() as conn:
        conn.execute(
            """INSERT INTO projects (id, title, created_at, updated_at, status, source_kind, source_url,
                   media_info_json, duration, has_video, source_meta_json)
               VALUES (?, ?, ?, ?, 'new', 'url', ?, '{}', ?, ?, ?)""",
            (project_id, title, now, now, url, meta.get("duration"), int(has_video), json.dumps(meta)),
        )
    return get_project(db, project_id)


def set_project_media(db: Database, project_id: str, media: MediaInfo, meta: dict[str, Any] | None = None) -> None:
    """Record the retrieved media file of a URL project (and its metadata)."""
    with db.transaction() as conn:
        conn.execute(
            """UPDATE projects SET source_path = ?, source_size = ?, source_mtime_ns = ?, media_info_json = ?,
                   duration = ?, has_video = ?, source_meta_json = COALESCE(?, source_meta_json), updated_at = ?
               WHERE id = ?""",
            (media.path, media.size_bytes, media.mtime_ns, json.dumps(media.to_dict()), media.duration,
             int(media.has_video), json.dumps(meta) if meta is not None else None, utc_now(), project_id),
        )


def save_captions(db: Database, project_id: str, tracks: list[dict[str, Any]], source: str) -> None:
    now = utc_now()
    with db.transaction() as conn:
        conn.execute("DELETE FROM captions WHERE project_id = ?", (project_id,))
        for track in tracks:
            conn.execute(
                """INSERT INTO captions (project_id, language, name, kind, source, fetched_at, cues_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (project_id, track["language"], track.get("name"), track["kind"], source, now,
                 json.dumps(track["cues"])),
            )


def list_captions(db: Database, project_id: str, include_cues: bool = False) -> list[dict[str, Any]]:
    out = []
    for r in db.query("SELECT * FROM captions WHERE project_id = ? ORDER BY kind DESC, id", (project_id,)):
        cues = json.loads(r["cues_json"])
        item = {"id": r["id"], "project_id": r["project_id"], "language": r["language"], "name": r["name"],
                "kind": r["kind"], "source": r["source"], "fetched_at": r["fetched_at"], "cue_count": len(cues)}
        if include_cues:
            item["cues"] = cues
        out.append(item)
    return out


def rename_project(db: Database, project_id: str, title: str) -> dict[str, Any]:
    with db.transaction() as conn:
        cur = conn.execute("UPDATE projects SET title = ?, updated_at = ? WHERE id = ?", (title, utc_now(), project_id))
        if cur.rowcount == 0:
            raise NotFound("Project not found.", data={"project_id": project_id})
    return get_project(db, project_id)


def set_project_status(conn, project_id: str, status: str, *, language: str | None = None, job_id: str | None = None) -> None:
    assert status in PROJECT_STATUSES
    sets = ["status = ?", "updated_at = ?"]
    params: list[Any] = [status, utc_now()]
    if language is not None:
        sets.append("language = ?")
        params.append(language)
    if job_id is not None:
        sets.append("last_job_id = ?")
        params.append(job_id)
    params.append(project_id)
    conn.execute(f"UPDATE projects SET {', '.join(sets)} WHERE id = ?", params)


def delete_project(db: Database, project_id: str) -> list[str]:
    """Deletes the project (cascades to jobs and transcript). Returns the ids of
    the deleted jobs so the caller can remove their working directories."""
    with db.transaction() as conn:
        job_ids = [r[0] for r in conn.execute("SELECT id FROM jobs WHERE project_id = ?", (project_id,))]
        cur = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        if cur.rowcount == 0:
            raise NotFound("Project not found.", data={"project_id": project_id})
    return job_ids


# --------------------------------------------------------------------------- jobs


def create_job(db: Database, project_id: str, mode: str, options: dict[str, Any], kind: str = "transcribe") -> dict[str, Any]:
    job_id = new_id()
    now = utc_now()
    with db.transaction() as conn:
        if conn.execute("SELECT 1 FROM projects WHERE id = ?", (project_id,)).fetchone() is None:
            raise NotFound("Project not found.", data={"project_id": project_id})
        conn.execute(
            """INSERT INTO jobs (id, project_id, status, mode, options_json, created_at, updated_at, kind,
                   queue_position)
               VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?)""",
            (job_id, project_id, mode, json.dumps(options), now, now, kind, _queue_end(conn)),
        )
        for stage in JOB_KIND_STAGES[kind]:
            conn.execute(
                "INSERT INTO job_stages (job_id, stage, ordinal, status) VALUES (?, ?, ?, 'pending')",
                (job_id, stage, STAGE_ORDINALS[stage]),
            )
        set_project_status(conn, project_id, "queued", job_id=job_id)
    return get_job(db, job_id)


def _elapsed(started: str | None, finished: str | None) -> float | None:
    start = parse_utc(started)
    if start is None:
        return None
    end = parse_utc(finished) or parse_utc(utc_now())
    return max(0.0, (end - start).total_seconds())


def _stage_dict(row, job_status: str) -> dict[str, Any]:
    done, total = row["progress_done"], row["progress_total"]
    fraction = None
    if done is not None and total:
        fraction = max(0.0, min(1.0, done / total))
    elapsed = _elapsed(row["started_at"], row["finished_at"])
    eta = None
    if row["status"] == "running" and job_status == "running" and fraction is not None and elapsed:
        baseline = row["baseline_done"] or 0.0
        measured = (done or 0.0) - baseline
        if total and measured / total >= _ETA_MIN_FRACTION and elapsed >= _ETA_MIN_SECONDS and measured > 0:
            rate = measured / elapsed
            eta = max(0.0, (total - done) / rate)
    return {
        "stage": row["stage"],
        "label": STAGE_LABELS.get(row["stage"], row["stage"]),
        "status": row["status"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "elapsed_seconds": elapsed,
        "progress": {"done": done, "total": total, "unit": row["progress_unit"]} if done is not None else None,
        "fraction": fraction,
        "eta_seconds": eta,
        "detail": row["detail"],
    }


def _job_row_to_dict(db: Database, row) -> dict[str, Any]:
    stages = db.query("SELECT * FROM job_stages WHERE job_id = ? ORDER BY ordinal", (row["id"],))
    project = db.query_one("SELECT title FROM projects WHERE id = ?", (row["project_id"],))
    return {
        "id": row["id"],
        "project_id": row["project_id"],
        "project_title": project["title"] if project else None,
        "kind": row["kind"],
        "status": row["status"],
        "mode": row["mode"],
        "options": json.loads(row["options_json"]),
        "effective": json.loads(row["effective_json"]) if row["effective_json"] else None,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "elapsed_seconds": _elapsed(row["started_at"], row["finished_at"]),
        "attempt": row["attempt"],
        "current_stage": row["current_stage"],
        "cancel_requested": bool(row["cancel_requested"]),
        "error": {"code": row["error_code"], "message": row["error_message"]} if row["error_code"] else None,
        "stages": [_stage_dict(s, row["status"]) for s in stages],
        # 1 = runs next; None unless queued
        "queue_position": _queue_rank(db, row) if row["status"] == "queued" else None,
    }


def _queue_rank(db: Database, row) -> int:
    ahead = db.query_one(
        """SELECT COUNT(*) FROM jobs WHERE status = 'queued'
           AND (queue_position < ? OR (queue_position = ? AND created_at < ?))""",
        (row["queue_position"], row["queue_position"], row["created_at"]))[0]
    return int(ahead) + 1


def get_job(db: Database, job_id: str) -> dict[str, Any]:
    row = db.query_one("SELECT * FROM jobs WHERE id = ?", (job_id,))
    if row is None:
        raise NotFound("Job not found.", data={"job_id": job_id})
    return _job_row_to_dict(db, row)


def get_job_row(db: Database, job_id: str):
    row = db.query_one("SELECT * FROM jobs WHERE id = ?", (job_id,))
    if row is None:
        raise NotFound("Job not found.", data={"job_id": job_id})
    return row


def list_jobs(
    db: Database, *, statuses: Iterable[str] | None = None, project_id: str | None = None, limit: int = 100
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM jobs"
    clauses, params = [], []
    if statuses:
        statuses = list(statuses)
        clauses.append(f"status IN ({','.join('?' * len(statuses))})")
        params.extend(statuses)
    if project_id:
        clauses.append("project_id = ?")
        params.append(project_id)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(int(limit))
    return [_job_row_to_dict(db, r) for r in db.query(sql, params)]


def _queue_end(conn) -> float:
    return float(conn.execute("SELECT COALESCE(MAX(queue_position), 0) + 1 FROM jobs").fetchone()[0])


def _queue_front(conn) -> float:
    row = conn.execute("SELECT MIN(queue_position) FROM jobs WHERE status = 'queued'").fetchone()[0]
    return (float(row) if row is not None else _queue_end(conn)) - 1


def next_queued_job(db: Database) -> str | None:
    row = db.query_one("SELECT id FROM jobs WHERE status = 'queued' ORDER BY queue_position, created_at LIMIT 1")
    return row["id"] if row else None


def queued_job_ids(db: Database) -> list[str]:
    """Queued jobs in the order they will run."""
    return [r["id"] for r in db.query("SELECT id FROM jobs WHERE status = 'queued' ORDER BY queue_position, created_at")]


def reorder_queue(db: Database, job_ids: list[str]) -> list[str]:
    """Run the queued jobs in this order. The list must name exactly the
    queued jobs (the app's view of the queue may be out of date)."""
    with db.transaction() as conn:
        current = [r["id"] for r in conn.execute(
            "SELECT id FROM jobs WHERE status = 'queued' ORDER BY queue_position, created_at")]
        if sorted(current) != sorted(job_ids) or len(set(job_ids)) != len(job_ids):
            raise InvalidJobState("The queue changed meanwhile. Reload it and try again.", data={"queued": current})
        start = _queue_front(conn) - len(job_ids)
        for i, job_id in enumerate(job_ids):
            conn.execute("UPDATE jobs SET queue_position = ? WHERE id = ?", (start + i, job_id))
    return queued_job_ids(db)


def mark_job_running(db: Database, job_id: str, pid: int) -> int:
    now = utc_now()
    with db.transaction() as conn:
        row = conn.execute("SELECT attempt, project_id, started_at FROM jobs WHERE id = ?", (job_id,)).fetchone()
        attempt = row["attempt"] + 1
        conn.execute(
            """UPDATE jobs SET status = 'running', worker_pid = ?, attempt = ?, started_at = ?,
                   finished_at = NULL, updated_at = ?, error_code = NULL, error_message = NULL
               WHERE id = ?""",
            (pid, attempt, now, now, job_id),
        )
        set_project_status(conn, row["project_id"], "processing")
    return attempt


def finish_job(
    db: Database,
    job_id: str,
    status: str,
    *,
    error_code: str | None = None,
    error_message: str | None = None,
) -> None:
    assert status in ("completed", "failed", "cancelled", "interrupted")
    now = utc_now()
    project_status = {"completed": "ready", "failed": "failed", "cancelled": "cancelled", "interrupted": "interrupted"}[status]
    with db.transaction() as conn:
        row = conn.execute("SELECT project_id, status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return
        conn.execute(
            """UPDATE jobs SET status = ?, finished_at = ?, updated_at = ?, worker_pid = NULL,
                   error_code = ?, error_message = ?
               WHERE id = ?""",
            (status, now, now, error_code, error_message, job_id),
        )
        stage_status = {"completed": None, "failed": "failed", "cancelled": "cancelled", "interrupted": "cancelled"}[status]
        if stage_status:
            conn.execute(
                "UPDATE job_stages SET status = ?, finished_at = ? WHERE job_id = ? AND status = 'running'",
                (stage_status, now, job_id),
            )
        # A project that already has a transcript from an earlier job stays usable.
        has_transcript = conn.execute("SELECT 1 FROM segments WHERE project_id = ? LIMIT 1", (row["project_id"],)).fetchone()
        if status != "completed" and has_transcript:
            project_status = "ready"
        set_project_status(conn, row["project_id"], project_status)


def request_cancel(db: Database, job_id: str) -> None:
    db.execute("UPDATE jobs SET cancel_requested = 1, updated_at = ? WHERE id = ?", (utc_now(), job_id))


def requeue_job(db: Database, job_id: str, *, front: bool = False) -> None:
    """Queue a stopped job again: at the end of the queue, or (work that was
    only interrupted) at its front."""
    now = utc_now()
    with db.transaction() as conn:
        row = conn.execute("SELECT project_id FROM jobs WHERE id = ?", (job_id,)).fetchone()
        position = _queue_front(conn) if front else _queue_end(conn)
        conn.execute(
            """UPDATE jobs SET status = 'queued', cancel_requested = 0, error_code = NULL,
                   error_message = NULL, finished_at = NULL, updated_at = ?, queue_position = ? WHERE id = ?""",
            (now, position, job_id),
        )
        conn.execute(
            "UPDATE job_stages SET status = 'pending' WHERE job_id = ? AND status IN ('failed','cancelled','running')",
            (job_id,),
        )
        set_project_status(conn, row["project_id"], "queued", job_id=job_id)


def set_effective(db: Database, job_id: str, effective: dict[str, Any]) -> None:
    db.execute("UPDATE jobs SET effective_json = ?, updated_at = ? WHERE id = ?", (json.dumps(effective), utc_now(), job_id))


# --------------------------------------------------------------------------- stages


def start_stage(db: Database, job_id: str, stage: str, *, unit: str | None = None, total: float | None = None,
                done: float | None = None, detail: str | None = None) -> None:
    now = utc_now()
    with db.transaction() as conn:
        # jobs created by an older engine lack the stages added since
        conn.execute(
            "INSERT OR IGNORE INTO job_stages (job_id, stage, ordinal, status) VALUES (?, ?, ?, 'pending')",
            (job_id, stage, STAGE_ORDINALS.get(stage, len(STAGES))),
        )
        conn.execute(
            """UPDATE job_stages SET status = 'running', started_at = ?, finished_at = NULL,
                   progress_unit = ?, progress_total = ?, progress_done = ?, baseline_done = ?, detail = ?
               WHERE job_id = ? AND stage = ?""",
            (now, unit, total, done, done or 0.0, detail, job_id, stage),
        )
        conn.execute("UPDATE jobs SET current_stage = ?, updated_at = ? WHERE id = ?", (stage, now, job_id))


def update_stage(db: Database, job_id: str, stage: str, *, done: float | None = None, total: float | None = None,
                 detail: str | None = None) -> None:
    sets, params = [], []
    if done is not None:
        sets.append("progress_done = ?")
        params.append(done)
    if total is not None:
        sets.append("progress_total = ?")
        params.append(total)
    if detail is not None:
        sets.append("detail = ?")
        params.append(detail)
    if not sets:
        return
    params.extend([job_id, stage])
    db.execute(f"UPDATE job_stages SET {', '.join(sets)} WHERE job_id = ? AND stage = ?", params)


def finish_stage(db: Database, job_id: str, stage: str, status: str = "completed", detail: str | None = None) -> None:
    now = utc_now()
    if detail is None:
        db.execute(
            "UPDATE job_stages SET status = ?, finished_at = ? WHERE job_id = ? AND stage = ?",
            (status, now, job_id, stage),
        )
    else:
        db.execute(
            "UPDATE job_stages SET status = ?, finished_at = ?, detail = ? WHERE job_id = ? AND stage = ?",
            (status, now, detail, job_id, stage),
        )


# --------------------------------------------------------------------------- audio & chunks


def save_audio_asset(db: Database, job_id: str, path: str, sample_rate: int, samples: int, duration: float,
                     source_size: int, source_mtime_ns: int, report: dict[str, Any]) -> None:
    db.execute(
        """INSERT OR REPLACE INTO audio_assets
               (job_id, path, sample_rate, samples, duration, source_size, source_mtime_ns, report_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (job_id, path, sample_rate, samples, duration, source_size, source_mtime_ns, json.dumps(report), utc_now()),
    )


def get_audio_asset(db: Database, job_id: str):
    return db.query_one("SELECT * FROM audio_assets WHERE job_id = ?", (job_id,))


def delete_audio_asset(db: Database, job_id: str) -> None:
    db.execute("DELETE FROM audio_assets WHERE job_id = ?", (job_id,))


def save_chunk_plan(db: Database, job_id: str, chunks: list[Chunk]) -> None:
    with db.transaction() as conn:
        conn.execute("DELETE FROM chunks WHERE job_id = ?", (job_id,))
        conn.executemany(
            """INSERT INTO chunks (job_id, idx, owned_start, owned_end, audio_start, audio_end, cut_energy_db, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')""",
            [(job_id, c.index, c.owned_start, c.owned_end, c.audio_start, c.audio_end, c.cut_energy_db) for c in chunks],
        )


def load_chunks(db: Database, job_id: str) -> list[tuple[Chunk, Any]]:
    rows = db.query("SELECT * FROM chunks WHERE job_id = ? ORDER BY idx", (job_id,))
    return [
        (
            Chunk(r["idx"], r["owned_start"], r["owned_end"], r["audio_start"], r["audio_end"], r["cut_energy_db"]),
            r,
        )
        for r in rows
    ]


def complete_chunk(db: Database, job_id: str, index: int, result: dict[str, Any], processing_seconds: float) -> None:
    db.execute(
        """UPDATE chunks SET status = 'completed', result_json = ?, language = ?, language_probability = ?,
               processing_seconds = ?, completed_at = ?
           WHERE job_id = ? AND idx = ?""",
        (
            json.dumps(result),
            result.get("language"),
            result.get("language_probability"),
            processing_seconds,
            utc_now(),
            job_id,
            index,
        ),
    )


# --------------------------------------------------------------------------- transcript


def replace_transcript(db: Database, project_id: str, job_id: str, segments: list[TSegment],
                       language: str | None, speakers: list[dict[str, Any]] | None = None) -> int:
    """Atomically replace the project's transcript with a new AI result.

    Speakers come from :func:`transcript.build.speakers_in_order`. The result
    is also stored, unchanged, as the project's original transcript; user
    corrections and their history from earlier results are discarded.
    """
    now = utc_now()
    with db.transaction() as conn:
        conn.execute("DELETE FROM edits WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM segments WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM speakers WHERE project_id = ?", (project_id,))
        ids: dict[int, int] = {}
        for sp in speakers or []:
            cur = conn.execute(
                """INSERT INTO speakers (project_id, label, name, origin, cluster_index, color_index, created_at)
                   VALUES (?, ?, NULL, 'diarization', ?, ?, ?)""",
                (project_id, sp["label"], sp["index"], sp.get("color_index", 0), now),
            )
            ids[sp["index"]] = cur.lastrowid
        for ordinal, seg in enumerate(segments):
            speaker_id = ids.get(seg.speaker) if seg.speaker is not None else None
            flags = []
            for flag in seg.flags:
                flag = dict(flag)
                if "other_speakers" in flag:
                    flag["other_speakers"] = [ids[o] for o in flag["other_speakers"] if o in ids]
                flags.append(flag)
            cur = conn.execute(
                """INSERT INTO segments (project_id, job_id, ordinal, start, end, original_text, text, language,
                       avg_logprob, no_speech_prob, compression_ratio, temperature, chunk_index, speaker_id,
                       original_speaker_id, speaker_confidence, flags_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    project_id, job_id, ordinal, seg.start, seg.end, seg.text, seg.text, seg.language,
                    seg.avg_logprob, seg.no_speech_prob, seg.compression_ratio, seg.temperature, seg.chunk_index,
                    speaker_id, speaker_id, seg.speaker_confidence, json.dumps(flags),
                ),
            )
            segment_id = cur.lastrowid
            if seg.words:
                conn.executemany(
                    """INSERT INTO words (segment_id, ordinal, start, end, text, probability, timing_source,
                           speaker_id, whisper_start, whisper_end, alignment_score, original_text)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (segment_id, i, w.start, w.end, w.text, w.probability, w.timing_source,
                         ids.get(w.speaker) if w.speaker is not None else None,
                         w.whisper_start, w.whisper_end, w.alignment_score, w.text)
                        for i, w in enumerate(seg.words)
                    ],
                )
        conn.execute(
            """UPDATE projects SET language = ?, updated_at = ?, transcript_revision = transcript_revision + 1
               WHERE id = ?""",
            (language, now, project_id),
        )
        snapshot = _transcript_payload(conn, project_id, include_words=True)
        conn.execute(
            """INSERT INTO original_transcripts (project_id, job_id, created_at, content_json) VALUES (?, ?, ?, ?)
               ON CONFLICT(project_id) DO UPDATE SET job_id = excluded.job_id, created_at = excluded.created_at,
                   content_json = excluded.content_json""",
            (project_id, job_id, now, json.dumps({"language": language, **snapshot})),
        )
    return len(segments)


def _speaker_dict(row, stats: dict[int, tuple[int, float, float | None]]) -> dict[str, Any]:
    count, seconds, first = stats.get(row["id"], (0, 0.0, None))
    return {
        "id": row["id"],
        "label": row["label"],
        "name": row["name"],
        "display_name": row["name"] or row["label"],
        "origin": row["origin"],
        "color_index": row["color_index"],
        "merged_into": row["merged_into"],
        "segment_count": count,
        "speaking_seconds": round(seconds, 3),
        "first_start": first,
    }


def _segment_dict(r) -> dict[str, Any]:
    return {
        "id": r["id"],
        "start": r["start"],
        "end": r["end"],
        "text": r["text"],
        "original_text": r["original_text"],
        "language": r["language"],
        "avg_logprob": r["avg_logprob"],
        "no_speech_prob": r["no_speech_prob"],
        "compression_ratio": r["compression_ratio"],
        "chunk_index": r["chunk_index"],
        "speaker_id": r["speaker_id"],
        "original_speaker_id": r["original_speaker_id"],
        "speaker_confidence": r["speaker_confidence"],
        "text_edited": bool(r["text_edited"]),
        "speaker_edited": bool(r["speaker_edited"]),
        "flags": (flags := json.loads(r["flags_json"])),
        "review_priority": review_priority(flags),
    }


def _word_dict(w) -> dict[str, Any]:
    return {
        "start": w["start"],
        "end": w["end"],
        "text": w["text"],
        "probability": w["probability"],
        "timing_source": w["timing_source"],
        "speaker_id": w["speaker_id"],
    }


def _transcript_payload(conn, project_id: str, include_words: bool) -> dict[str, Any]:
    rows = conn.execute("SELECT * FROM segments WHERE project_id = ? ORDER BY start, id", (project_id,)).fetchall()
    words_by_segment: dict[int, list[dict[str, Any]]] = {}
    if include_words and rows:
        for w in conn.execute(
            """SELECT w.* FROM words w JOIN segments s ON s.id = w.segment_id
               WHERE s.project_id = ? ORDER BY w.segment_id, w.ordinal""",
            (project_id,),
        ):
            words_by_segment.setdefault(w["segment_id"], []).append(_word_dict(w))
    stats: dict[int, tuple[int, float, float | None]] = {}
    segments = []
    for ordinal, r in enumerate(rows):
        seg = {**_segment_dict(r), "ordinal": ordinal}
        if include_words:
            seg["words"] = words_by_segment.get(r["id"], [])
        segments.append(seg)
        if r["speaker_id"] is not None:
            count, seconds, first = stats.get(r["speaker_id"], (0, 0.0, None))
            stats[r["speaker_id"]] = (count + 1, seconds + max(0.0, r["end"] - r["start"]),
                                      r["start"] if first is None else first)
    everyone = [
        _speaker_dict(sp, stats)
        for sp in conn.execute("SELECT * FROM speakers WHERE project_id = ? ORDER BY id", (project_id,))
    ]
    # transcript order; speakers without segments (user-created, not used yet) last
    everyone.sort(key=lambda sp: (sp["first_start"] is None, sp["first_start"] or 0.0, sp["id"]))
    return {
        "speakers": [sp for sp in everyone if sp["merged_into"] is None],
        # merged away by the user; still referenced as the AI's original speaker
        "merged_speakers": [sp for sp in everyone if sp["merged_into"] is not None],
        "segments": segments,
    }


def history_state(conn, project_id: str) -> dict[str, Any]:
    undo = conn.execute(
        "SELECT summary FROM edits WHERE project_id = ? AND undone = 0 ORDER BY id DESC LIMIT 1", (project_id,)
    ).fetchone()
    redo = conn.execute(
        "SELECT summary FROM edits WHERE project_id = ? AND undone = 1 ORDER BY id ASC LIMIT 1", (project_id,)
    ).fetchone()
    count = conn.execute("SELECT COUNT(*) FROM edits WHERE project_id = ? AND undone = 0", (project_id,)).fetchone()[0]
    return {
        "can_undo": undo is not None,
        "can_redo": redo is not None,
        "undo_summary": undo["summary"] if undo else None,
        "redo_summary": redo["summary"] if redo else None,
        "edit_count": count,
    }


def get_transcript(db: Database, project_id: str, include_words: bool = False) -> dict[str, Any]:
    project = get_project(db, project_id)
    with db.snapshot() as conn:
        payload = _transcript_payload(conn, project_id, include_words)
        history = history_state(conn, project_id)
        revision = conn.execute("SELECT transcript_revision FROM projects WHERE id = ?", (project_id,)).fetchone()[0]
        first = conn.execute(
            "SELECT job_id FROM segments WHERE project_id = ? AND job_id IS NOT NULL LIMIT 1", (project_id,)
        ).fetchone()
        original = conn.execute("SELECT job_id FROM original_transcripts WHERE project_id = ?",
                                (project_id,)).fetchone()
    # Settings of the job that produced this transcript (not necessarily the
    # project's latest job, which may have failed), also for an empty one.
    transcript_job_id = first["job_id"] if first else (original["job_id"] if original else None)
    effective = None
    if transcript_job_id:
        job_row = db.query_one("SELECT effective_json FROM jobs WHERE id = ?", (transcript_job_id,))
        if job_row and job_row["effective_json"]:
            effective = json.loads(job_row["effective_json"])
    return {
        "project_id": project_id,
        "language": project["language"],
        "job_id": transcript_job_id,
        "effective": effective,
        "revision": revision,
        "history": history,
        **payload,
    }


def get_original_transcript(db: Database, project_id: str) -> dict[str, Any] | None:
    row = db.query_one("SELECT * FROM original_transcripts WHERE project_id = ?", (project_id,))
    if row is None:
        return None
    return {"project_id": project_id, "job_id": row["job_id"], "created_at": row["created_at"],
            **json.loads(row["content_json"])}


# --------------------------------------------------------------------------- settings


def get_setting(db: Database, key: str, default: Any = None) -> Any:
    row = db.query_one("SELECT value_json FROM settings WHERE key = ?", (key,))
    return json.loads(row["value_json"]) if row else default


def set_setting(db: Database, key: str, value: Any) -> None:
    db.execute(
        "INSERT INTO settings (key, value_json) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
        (key, json.dumps(value)),
    )


def load_segments(db: Database, project_id: str) -> list[tuple[int, Any, bool]]:
    """The current transcript as engine segments: ``[(segment_id, TSegment,
    text_edited)]`` in time order (words keep their database speaker ids)."""
    from ..transcript.model import TSegment, TWord

    out = []
    for r in db.query("SELECT * FROM segments WHERE project_id = ? ORDER BY start, id", (project_id,)):
        words = [
            TWord(w["start"], w["end"], w["text"], w["probability"], w["timing_source"], w["whisper_start"],
                  w["whisper_end"], w["alignment_score"], w["speaker_id"])
            for w in db.query("SELECT * FROM words WHERE segment_id = ? ORDER BY ordinal", (r["id"],))
        ]
        seg = TSegment(r["start"], r["end"], r["text"], words, r["language"], r["avg_logprob"], r["no_speech_prob"],
                       r["compression_ratio"], r["temperature"], r["chunk_index"], json.loads(r["flags_json"]),
                       r["speaker_id"], r["speaker_confidence"])
        out.append((r["id"], seg, bool(r["text_edited"])))
    return out
