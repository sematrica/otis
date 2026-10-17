"""Stage 3 corrections and AI suggestions through the RPC methods (no models).

The transcript is written directly; the re-check flags and the stored
suggestions have the shapes the re-check pass and an AI review produce. These
tests check the guarantees: every change is one undo step, a user's choice is
recorded, nothing an AI suggests is applied until accepted, outdated
suggestions are never applied, and a decision cannot be overwritten.
"""

from __future__ import annotations

import pytest

from verbatim_engine.errors import InvalidJobState, InvalidParams
from verbatim_engine.paths import EnginePaths
from verbatim_engine.rpc.service import EngineService
from verbatim_engine.storage import repo
from verbatim_engine.transcript.model import TSegment, TWord
from verbatim_engine.util.ids import utc_now


def _segment(start: float, text: str, flags=None) -> TSegment:
    words = [TWord(start + 0.4 * i, start + 0.4 * i + 0.3, " " + w, 0.9) for i, w in enumerate(text.split())]
    return TSegment(words[0].start, words[-1].end, text, words, "en", flags=list(flags or []))


ALTERNATIVES = {"type": "alternatives", "severity": "medium", "margin": 1.2, "candidates": [
    {"source": "first_pass", "text": "daddy and a new jersey", "score": -40.1},
    {"source": "context", "text": "Diane in New Jersey", "score": -39.0}]}


@pytest.fixture()
def service(tmp_path, jfk_path):
    svc = EngineService(EnginePaths.create(tmp_path / "data"), lambda *_: None)
    svc.start()
    project = svc.projects_create({"path": str(jfk_path), "title": "Corrections"})
    repo.replace_transcript(svc.db, project["id"], None, [
        _segment(0.0, "hello there how are you"),
        _segment(3.0, "daddy and a new jersey", [ALTERNATIVES]),
        _segment(6.0, "we moved to kubernetes last week", [{"type": "marked_uncertain", "note": "check"}]),
    ], "en", [])
    svc.project_id = project["id"]
    yield svc
    svc.shutdown()


def _segments(svc) -> list[dict]:
    return repo.get_transcript(svc.db, svc.project_id, include_words=True)["segments"]


def test_choosing_a_recorded_wording_is_one_undoable_step(service):
    call = service.methods()
    seg = _segments(service)[1]
    assert seg["review_priority"] == "medium"
    result = call["transcript.chooseWording"]({"project_id": service.project_id, "segment_id": seg["id"],
                                               "source": "context"})
    chosen = next(s for s in result["transcript"]["segments"] if s["id"] == seg["id"])
    assert chosen["text"].strip() == "Diane in New Jersey" and chosen["text_edited"]
    flag = next(f for f in chosen["flags"] if f["type"] == "alternatives")
    assert flag["resolved"] and flag["chosen_by_user"] == "context"
    assert chosen["review_priority"] is None
    assert [w["text"].strip() for w in chosen["words"]] == ["Diane", "in", "New", "Jersey"]
    assert result["summary"].startswith("Use alternative wording")
    with pytest.raises(InvalidParams):
        call["transcript.chooseWording"]({"project_id": service.project_id, "segment_id": seg["id"],
                                          "source": "nonexistent"})
    undone = call["transcript.undo"]({"project_id": service.project_id})["transcript"]
    restored = next(s for s in undone["segments"] if s["id"] == seg["id"])
    assert restored["text"].strip() == "daddy and a new jersey" and restored["review_priority"] == "medium"


def test_marking_reviewed_clears_automatic_doubts_but_not_the_users_own(service):
    call = service.methods()
    first, flagged, uncertain = _segments(service)
    result = call["transcript.markReviewed"]({"project_id": service.project_id,
                                              "segment_ids": [flagged["id"], uncertain["id"]]})
    after = {s["id"]: s for s in result["transcript"]["segments"]}
    assert after[flagged["id"]]["review_priority"] is None
    assert after[uncertain["id"]]["review_priority"] == "high"  # a user's mark is cleared only by the user
    unmarked = call["transcript.markReviewed"]({"project_id": service.project_id, "segment_ids": [flagged["id"]],
                                                "reviewed": False})["transcript"]
    assert next(s for s in unmarked["segments"] if s["id"] == flagged["id"])["review_priority"] == "medium"


def test_removing_a_passage_is_undoable(service):
    call = service.methods()
    segments = _segments(service)
    removed = call["transcript.deleteSegment"]({"project_id": service.project_id, "segment_id": segments[0]["id"]})
    assert [s["id"] for s in removed["transcript"]["segments"]] == [s["id"] for s in segments[1:]]
    undone = call["transcript.undo"]({"project_id": service.project_id})["transcript"]
    assert [s["text"] for s in undone["segments"]] == [s["text"] for s in segments]
    assert [w["text"] for w in undone["segments"][0]["words"]] == [w["text"] for w in segments[0]["words"]]


def test_recheck_needs_the_transcription_that_produced_the_transcript(service):
    seg = _segments(service)[0]
    with pytest.raises(InvalidJobState):
        service.methods()["transcript.recheck"]({"project_id": service.project_id, "segment_ids": [seg["id"]]})


def _suggest(svc, segment: dict, kind: str, suggested: str | None, reason: str = "") -> int:
    run = svc.db.execute(
        """INSERT INTO llm_runs (project_id, provider, model, endpoint, cloud, status, transcript_revision,
                                 started_at) VALUES (?, 'ollama', 'test-model', 'http://127.0.0.1:11434', 0,
                                 'completed', 1, ?)""", (svc.project_id, utc_now())).lastrowid
    return svc.db.execute(
        """INSERT INTO suggestions (project_id, run_id, segment_id, kind, original_text, suggested_text, reason,
                                    status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
        (svc.project_id, run, segment["id"], kind, segment["text"], suggested, reason, utc_now())).lastrowid


def test_accepting_suggestions_applies_them_traceably(service):
    call = service.methods()
    first, _, third = _segments(service)
    punctuation = _suggest(service, first, "punctuation", "Hello there, how are you?")
    suspicious = _suggest(service, third, "suspicious", None, "kubernetes may be misheard")
    assert repo.get_transcript(service.db, service.project_id)["revision"] == 1  # nothing applied yet

    result = call["suggestions.accept"]({"suggestion_id": punctuation})
    assert result["changed"] and result["suggestion"]["status"] == "accepted" and result["job"] is None
    assert result["summary"] == "Apply AI punctuation suggestion at 00:00:00 (ollama / test-model)"
    assert next(s for s in result["transcript"]["segments"] if s["id"] == first["id"])["text"].strip() == \
        "Hello there, how are you?"
    with pytest.raises(InvalidParams):  # a decision stands
        call["suggestions.reject"]({"suggestion_id": punctuation})

    marked = call["suggestions.accept"]({"suggestion_id": suspicious})
    flags = next(s for s in marked["transcript"]["segments"] if s["id"] == third["id"])["flags"]
    note = next(f for f in flags if f["type"] == "marked_uncertain")["note"]
    assert note == "AI review (ollama / test-model): kubernetes may be misheard"
    assert call["transcript.undo"]({"project_id": service.project_id})["changed"]


def test_a_suggestion_for_a_changed_passage_is_never_applied(service):
    call = service.methods()
    first = _segments(service)[0]
    suggestion = _suggest(service, first, "terminology", "hello there how are you")
    call["transcript.editText"]({"project_id": service.project_id, "segment_id": first["id"], "text": "Hi there"})
    result = call["suggestions.accept"]({"suggestion_id": suggestion})
    assert not result["changed"] and result["suggestion"]["status"] == "outdated"
    assert _segments(service)[0]["text"].strip() == "Hi there"


def test_a_remote_endpoint_needs_explicit_permission(service):
    with pytest.raises(InvalidParams) as refused:
        service.methods()["llm.review"]({"project_id": service.project_id, "provider": "openai",
                                         "model": "example-model", "base_url": "https://api.openai.example/v1",
                                         "api_key": "placeholder"})
    assert refused.value.data == {"remote": True}
    assert service.db.query_one("SELECT COUNT(*) AS n FROM llm_runs")["n"] == 0  # nothing was started or sent
