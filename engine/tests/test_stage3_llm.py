"""AI review against a real local language model (Ollama).

Needs ``VERBATIM_TEST_OLLAMA_MODEL`` (a pulled model, e.g. a small instruct
model) and optionally ``VERBATIM_TEST_OLLAMA_URL``. The assertions check the
guarantees, not the model's judgement: runs are recorded with every exchange,
stored suggestions are either validated or marked invalid with the reason,
nothing changes until a suggestion is accepted, and accepting is undoable.
The OpenAI provider is exercised against Ollama's OpenAI-compatible API.
"""

from __future__ import annotations

import os

import pytest

from verbatim_engine.asr.merge import normalize_token
from verbatim_engine.llm.providers import OpenAIProvider
from verbatim_engine.paths import EnginePaths
from verbatim_engine.rpc.service import EngineService
from verbatim_engine.storage import repo
from verbatim_engine.transcript.model import TSegment, TWord

pytestmark = pytest.mark.model

MODEL = os.environ.get("VERBATIM_TEST_OLLAMA_MODEL")
URL = os.environ.get("VERBATIM_TEST_OLLAMA_URL", "http://127.0.0.1:11434")

# Unpunctuated text with an inconsistent spelling, as a recogniser might produce it.
TEXTS = [
    "and so my fellow americans ask not what your country can do for you",
    "we moved the build server to kubernetes last week",
    "the cooper netties cluster now has three nodes and it works well",
    "thank you for watching",
]


@pytest.fixture()
def service(tmp_path, jfk_path):
    if not MODEL:
        pytest.skip("VERBATIM_TEST_OLLAMA_MODEL not set: a local Ollama model is required")
    svc = EngineService(EnginePaths.create(tmp_path / "data"), lambda *_: None)
    svc.start()
    yield svc
    svc.shutdown()


def _project_with_transcript(svc: EngineService, jfk_path) -> str:
    project = svc.projects_create({"path": str(jfk_path), "title": "Review"})
    segments = []
    t = 0.0
    for text in TEXTS:
        words = []
        for token in text.split():
            words.append(TWord(t, t + 0.3, " " + token, 0.9))
            t += 0.35
        segments.append(TSegment(words[0].start, words[-1].end, text, words, "en"))
        t += 1.0
    repo.replace_transcript(svc.db, project["id"], None, segments, "en", [])
    return project["id"]


def _words(text: str) -> list[str]:
    return [t for t in (normalize_token(w) for w in text.split()) if t]


def test_review_with_a_local_model_is_validated_traceable_and_reversible(service, jfk_path):
    project_id = _project_with_transcript(service, jfk_path)
    before = repo.get_transcript(service.db, project_id)
    run = service.methods()["llm.review"]({"project_id": project_id, "provider": "ollama", "model": MODEL,
                                           "base_url": URL})
    assert run["remote"] is False and run["status"] == "running"
    run = service.reviews.wait(run["id"], timeout=600)
    assert run["status"] == "completed", run["error"]
    detail = service.methods()["llm.run"]({"run_id": run["id"], "exchanges": True})
    assert detail["requests"] >= 1 and len(detail["exchanges"]) == detail["requests"]
    assert all(e["response"] for e in detail["exchanges"])  # the model's own answers are kept

    suggestions = service.methods()["suggestions.list"]({"project_id": project_id})["suggestions"]
    print(f"{len(suggestions)} suggestion(s):", [(s["kind"], s["status"], s["suggested_text"]) for s in suggestions])
    for s in suggestions:
        assert s["status"] in ("pending", "invalid")
        if s["status"] == "invalid":
            assert s["validation"]
        if s["kind"] == "punctuation" and s["status"] == "pending":
            assert _words(s["suggested_text"]) == _words(s["original_text"])  # words never change
    # nothing was applied
    after = repo.get_transcript(service.db, project_id)
    assert after["revision"] == before["revision"]
    assert [x["text"] for x in after["segments"]] == [x["text"] for x in before["segments"]]

    applicable = [s for s in suggestions if s["status"] == "pending" and s["kind"] in ("punctuation", "terminology")]
    if not applicable:
        pytest.skip("the model made no punctuation or spelling suggestion to apply (its choice)")
    chosen = applicable[0]
    result = service.methods()["suggestions.accept"]({"suggestion_id": chosen["id"]})
    assert result["changed"] and result["suggestion"]["status"] == "accepted"
    changed = next(x for x in result["transcript"]["segments"] if x["id"] == chosen["segment_id"])
    assert changed["text"].strip() == chosen["suggested_text"].strip()
    assert result["transcript"]["history"]["undo_summary"].startswith("Apply AI")
    undone = service.methods()["transcript.undo"]({"project_id": project_id})
    restored = next(x for x in undone["transcript"]["segments"] if x["id"] == chosen["segment_id"])
    assert restored["text"].strip() == chosen["original_text"].strip()


def test_openai_provider_speaks_the_chat_completions_protocol(service):
    """Ollama implements the OpenAI Chat Completions API locally."""
    provider = OpenAIProvider(MODEL, api_key=None, base_url=f"{URL}/v1")
    assert provider.remote is False
    assert MODEL in provider.list_models() or any(m.startswith(MODEL.split(":")[0]) for m in provider.list_models())
    from verbatim_engine.llm.review import SCHEMA, SYSTEM_PROMPT, collect

    batch = [{"id": 1, "text": TEXTS[0], "speaker_id": None}]
    parsed, exchange = provider.complete_json(SYSTEM_PROMPT, "Transcript:\n[1] Speaker 1: " + TEXTS[0], SCHEMA)
    assert isinstance(parsed, dict) and exchange.response
    for record in collect(parsed, batch):
        assert record["status"] in ("pending", "invalid")
