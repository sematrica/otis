"""Stage 4 full-text search (no models): the index follows every change to
the transcript, user text is never FTS5 syntax, and results carry what the
app needs to jump to the passage."""

from __future__ import annotations

import pytest

from verbatim_engine.errors import InvalidParams
from verbatim_engine.paths import EnginePaths
from verbatim_engine.rpc.service import EngineService
from verbatim_engine.search import fts_query, parse_terms
from verbatim_engine.storage import repo
from verbatim_engine.transcript.model import TSegment, TWord


def _segment(start: float, text: str, speaker: int | None = None, language: str = "en") -> TSegment:
    words = [TWord(start + 0.4 * i, start + 0.4 * i + 0.3, " " + w, 0.9, speaker=speaker)
             for i, w in enumerate(text.split())]
    return TSegment(words[0].start, words[-1].end, text, words, language, speaker=speaker)


@pytest.fixture()
def service(tmp_path, jfk_path):
    svc = EngineService(EnginePaths.create(tmp_path / "data"), lambda *_: None)
    svc.start()
    speakers = [{"label": "Speaker 1", "index": 0, "color_index": 0}, {"label": "Speaker 2", "index": 1, "color_index": 1}]
    first = svc.projects_create({"path": str(jfk_path), "title": "Bridges"})
    repo.replace_transcript(svc.db, first["id"], None, [
        _segment(0.0, "The café by the river opened in 1984.", 0),
        _segment(5.0, "We built the bridge over the river together.", 1),
        _segment(10.0, "Bridges need maintenance; ours got a new coat of paint.", 0),
    ], "en", speakers)
    second = svc.projects_create({"path": str(jfk_path), "title": "Rivers"})
    repo.replace_transcript(svc.db, second["id"], None, [
        _segment(0.0, "A river is not a bridge.", None),
        _segment(3.0, "東京の橋を渡りました", None, "ja"),
    ], "en", [])
    svc.ids = (first["id"], second["id"])
    yield svc
    svc.shutdown()


def _find(svc, query, **params):
    return svc.methods()["search.transcripts"]({"query": query, **params})


def test_words_match_as_prefixes_ignoring_case_and_diacritics(service):
    found = _find(service, "CAFE")
    assert [r["text"] for r in found["results"]] == ["The café by the river opened in 1984."]
    hit = found["results"][0]
    assert hit["text"][slice(*hit["highlights"][0])] == "café"
    assert hit["project_title"] == "Bridges" and hit["speaker"] == "Speaker 1" and hit["start"] == 0.0
    both = _find(service, "brid riv")  # every word must occur; prefixes match
    assert {r["text"] for r in both["results"]} == {"We built the bridge over the river together.",
                                                     "A river is not a bridge."}
    assert all(len(r["highlights"]) == 2 for r in both["results"])


def test_phrases_projects_and_speakers_narrow_the_search(service):
    first, second = service.ids
    assert [r["text"] for r in _find(service, '"the river"')["results"]] == [
        "The café by the river opened in 1984.", "We built the bridge over the river together."]
    assert [r["project_id"] for r in _find(service, "river", project_id=second)["results"]] == [second]
    speaker2 = next(sp["id"] for sp in repo.get_transcript(service.db, first)["speakers"] if sp["label"] == "Speaker 2")
    assert [r["text"] for r in _find(service, "river", speaker_id=speaker2)["results"]] == [
        "We built the bridge over the river together."]
    within = _find(service, "bridge", project_id=first)["results"]
    assert [r["start"] for r in within] == sorted(r["start"] for r in within)  # time order within a project


def test_user_text_is_never_query_syntax(service):
    for hostile in ['river" OR "x', "NEAR(river bridge)", "river*", "text:river", "-river", "^river", "(river"]:
        result = _find(service, hostile)  # no FTS5 syntax error
        assert all("river" in r["text"].lower() for r in result["results"]), hostile
    # an unclosed quote is an ordinary character, escaped for FTS5
    assert fts_query(parse_terms('a "b c'), False) == 'text : ("a"* """b"* "c"*)'
    with pytest.raises(InvalidParams):
        _find(service, '  ;; "" ')


def test_the_index_follows_corrections_undo_and_new_transcripts(service):
    first, second = service.ids
    call = service.methods()
    seg = repo.get_transcript(service.db, first)["segments"][1]
    call["transcript.editText"]({"project_id": first, "segment_id": seg["id"], "text": "We built the viaduct together."})
    assert not _find(service, "viaduct bridge")["results"]
    assert [r["segment_id"] for r in _find(service, "viaduct")["results"]] == [seg["id"]]
    # the AI's original wording can still be found on request
    original = _find(service, "bridge over", include_original=True)["results"]
    assert [r["segment_id"] for r in original] == [seg["id"]] and original[0]["in_original_only"]
    call["transcript.undo"]({"project_id": first})
    assert not _find(service, "viaduct")["results"]
    assert seg["id"] in [r["segment_id"] for r in _find(service, "bridge over")["results"]]
    call["transcript.deleteSegment"]({"project_id": first, "segment_id": seg["id"]})
    assert seg["id"] not in [r["segment_id"] for r in _find(service, "bridge")["results"]]
    repo.replace_transcript(service.db, second, None, [_segment(0.0, "Completely new words.")], "en", [])
    assert not _find(service, "river", project_id=second)["results"]
    assert _find(service, "completely")["results"][0]["project_id"] == second


def test_languages_without_spaces_are_searched_by_substring(service):
    found = _find(service, "橋")
    assert found["method"] == "substring"
    hit = found["results"][0]
    assert hit["text"][slice(*hit["highlights"][0])] == "橋"


def test_results_are_limited(service):
    result = _find(service, "the", limit=1)
    assert len(result["results"]) == 1 and result["truncated"]
