"""Stage 4 persistent processing queue (no models).

Order, reordering and the rules for resuming interrupted work are checked on
the database and the job manager. Where a job must actually be started, the
real worker process runs it; without an installed model it fails at once,
which is enough to see whether (and when) the manager started it.
"""

from __future__ import annotations

import time

import pytest

from verbatim_engine.errors import InvalidJobState
from verbatim_engine.jobs.manager import MAX_AUTO_ATTEMPTS, SETTING_AUTO_RESUME, SETTING_PAUSED, JobManager
from verbatim_engine.jobs.options import normalize_options
from verbatim_engine.paths import EnginePaths
from verbatim_engine.rpc.service import EngineService
from verbatim_engine.storage import repo
from verbatim_engine.storage.db import Database

OPTIONS = normalize_options({"mode": "fast", "model_id": "faster-whisper/small"})


@pytest.fixture()
def env(tmp_path, jfk_path):
    paths = EnginePaths.create(tmp_path / "data")
    db = Database(paths.db_path)
    service = EngineService(paths, lambda *_: None)  # projects need a probed source
    projects = [service.projects_create({"path": str(jfk_path), "title": f"P{i}"})["id"] for i in range(4)]
    service.db.close()
    yield paths, db, projects
    db.close()


def _job(db, project_id: str) -> str:
    return repo.create_job(db, project_id, "fast", OPTIONS)["id"]


def test_queued_jobs_run_in_queue_order_and_can_be_reordered(env):
    _paths, db, projects = env
    a, b, c = (_job(db, p) for p in projects[:3])
    assert repo.queued_job_ids(db) == [a, b, c] and repo.next_queued_job(db) == a
    assert [repo.get_job(db, j)["queue_position"] for j in (a, b, c)] == [1, 2, 3]
    assert repo.reorder_queue(db, [c, a, b]) == [c, a, b]
    assert repo.next_queued_job(db) == c
    d = _job(db, projects[3])  # new work goes to the end
    assert repo.queued_job_ids(db) == [c, a, b, d]
    with pytest.raises(InvalidJobState):
        repo.reorder_queue(db, [a, b, c])  # out of date: d is missing
    with pytest.raises(InvalidJobState):
        repo.reorder_queue(db, [a, a, b, d])
    assert repo.queued_job_ids(db) == [c, a, b, d]  # unchanged after a refused reorder


def test_interrupted_work_goes_to_the_front_and_manual_resume_to_the_end(env):
    paths, db, projects = env
    a, b, c = (_job(db, p) for p in projects[:3])
    repo.finish_job(db, a, "interrupted", error_code="interrupted", error_message="quit")
    repo.finish_job(db, c, "cancelled", error_code="cancelled", error_message="cancelled")
    repo.requeue_job(db, a, front=True)
    repo.requeue_job(db, c)
    assert repo.queued_job_ids(db) == [a, b, c]
    assert repo.get_job(db, a)["error"] is None


def test_resume_on_launch_only_continues_the_latest_job_of_a_project(env):
    paths, db, projects = env
    old = _job(db, projects[0])
    repo.finish_job(db, old, "interrupted", error_code="interrupted", error_message="quit")
    newer = _job(db, projects[0])
    repo.finish_job(db, newer, "completed")  # a newer transcript exists: never overwrite it
    latest = _job(db, projects[1])
    repo.finish_job(db, latest, "interrupted", error_code="interrupted", error_message="quit")
    stuck = _job(db, projects[2])
    db.execute("UPDATE jobs SET attempt = ? WHERE id = ?", (MAX_AUTO_ATTEMPTS, stuck))
    repo.finish_job(db, stuck, "interrupted", error_code="interrupted", error_message="quit")
    manager = JobManager(db, paths, lambda *_: None)
    assert manager.resume_interrupted() == [latest]
    assert repo.get_job(db, old)["status"] == "interrupted"
    assert repo.get_job(db, stuck)["status"] == "interrupted"
    assert repo.queued_job_ids(db) == [latest]


def test_resume_on_launch_can_be_turned_off(env):
    paths, db, projects = env
    job = _job(db, projects[0])
    repo.finish_job(db, job, "interrupted", error_code="interrupted", error_message="quit")
    repo.set_setting(db, SETTING_AUTO_RESUME, False)
    repo.set_setting(db, SETTING_PAUSED, True)  # nothing may start during this test
    manager = JobManager(db, paths, lambda *_: None)
    manager.start()
    try:
        assert repo.get_job(db, job)["status"] == "interrupted"
        assert manager.queue_state()["auto_resume"] is False
    finally:
        manager.shutdown()


def _wait(db, job_id, statuses, timeout=60.0):
    deadline = time.monotonic() + timeout
    while repo.get_job(db, job_id)["status"] not in statuses:
        assert time.monotonic() < deadline, repo.get_job(db, job_id)["status"]
        time.sleep(0.1)
    return repo.get_job(db, job_id)


def test_a_paused_queue_starts_nothing_until_resumed(env):
    paths, db, projects = env
    notes = []
    manager = JobManager(db, paths, lambda method, params: notes.append((method, params)))
    state = manager.pause()
    assert state["paused"] is True
    job = _job(db, projects[0])
    manager.start()
    try:
        time.sleep(2.5)  # the scheduler wakes every second
        assert repo.get_job(db, job)["status"] == "queued" and repo.get_job(db, job)["attempt"] == 0
        assert manager.queue_state()["queued"][0]["id"] == job
        manager.resume_queue()
        done = _wait(db, job, ("failed", "completed"))
        assert done["attempt"] == 1  # it was started (and failed: no model is installed here)
        assert done["error"]["code"] == "model_not_installed"
    finally:
        manager.shutdown()
    updates = [params for method, params in notes if method == "queue.updated"]
    assert updates[0]["paused"] is True and updates[-1]["paused"] is False
    assert repo.get_setting(db, SETTING_PAUSED) is False


def test_queue_state_survives_a_restart(env):
    paths, db, projects = env
    a, b = (_job(db, p) for p in projects[:2])
    manager = JobManager(db, paths, lambda *_: None)
    manager.pause()
    manager.reorder([b, a])
    again = JobManager(db, paths, lambda *_: None)  # a new engine process
    state = again.queue_state()
    assert state["paused"] is True and [j["id"] for j in state["queued"]] == [b, a]
