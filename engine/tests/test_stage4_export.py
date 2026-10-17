"""Stage 4 exports (no models): each format is parsed back and checked
against the transcript it came from."""

from __future__ import annotations

import csv
import io
import json
import re

import pytest

from verbatim_engine.export.subtitles import MAX_LINE, MAX_SECONDS
from verbatim_engine.paths import EnginePaths
from verbatim_engine.rpc.service import EngineService
from verbatim_engine.storage import repo
from verbatim_engine.transcript.model import TSegment, TWord

TEXTS = [
    (0, "Hello and welcome to the show. Today we talk about rivers, bridges and the people who build them."),
    (1, "=SUM(A1) is not a formula here, and <b>tags</b> & ampersands stay text."),
    (0, "Thanks. Let's start with the first bridge, which was finished in 1984 after eleven long years of work "
        "by more than two hundred people from the valley."),
]


def _segments() -> list[TSegment]:
    out, t = [], 0.0
    for speaker, text in TEXTS:
        words = []
        for token in text.split():
            words.append(TWord(t, t + 0.3, " " + token, 0.9, "alignment", speaker=speaker))
            t += 0.4
        seg = TSegment(words[0].start, words[-1].end, text, words, "en", speaker=speaker)
        out.append(seg)
        t += 1.0
    out[1].flags.append({"type": "low_confidence", "severity": "medium", "reasons": ["many_uncertain_words"]})
    return out


@pytest.fixture()
def service(tmp_path, jfk_path):
    svc = EngineService(EnginePaths.create(tmp_path / "data"), lambda *_: None)
    svc.start()
    project = svc.projects_create({"path": str(jfk_path), "title": "Rivers & Bridges"})
    speakers = [{"label": "Speaker 1", "index": 0, "color_index": 0}, {"label": "Speaker 2", "index": 1, "color_index": 1}]
    repo.replace_transcript(svc.db, project["id"], None, _segments(), "en", speakers)
    svc.project_id = project["id"]
    yield svc
    svc.shutdown()


def _export(svc, fmt: str, **options) -> str:
    result = svc.export_render({"project_id": svc.project_id, "format": fmt, "options": options})
    assert result["suggested_filename"] == f"Rivers & Bridges.{fmt}"
    return result["content"]


def _all_words() -> list[str]:
    return [w for _, text in TEXTS for w in text.split()]


TIME = r"(\d\d):(\d\d):(\d\d)[,.](\d\d\d)"


def _seconds(match) -> float:
    h, m, s, ms = (int(x) for x in match)
    return h * 3600 + m * 60 + s + ms / 1000


def _cues(text: str, sep: str) -> list[tuple[float, float, list[str]]]:
    cues = []
    for block in text.strip().split("\n\n"):
        lines = block.split("\n")
        for i, line in enumerate(lines):
            m = re.fullmatch(f"{TIME} --> {TIME}", line)
            if m:
                g = m.groups()
                cues.append((_seconds(g[:4]), _seconds(g[4:]), lines[i + 1:]))
                break
    return cues


def test_srt_cues_follow_the_words_and_subtitle_limits(service):
    srt = _export(service, "srt")
    blocks = srt.strip().split("\n\n")
    assert [b.split("\n")[0] for b in blocks] == [str(i) for i in range(1, len(blocks) + 1)]
    cues = _cues(srt, ",")
    words: list[str] = []
    for start, end, lines in cues:
        assert end > start and end - start <= MAX_SECONDS + 0.4
        assert len(lines) <= 2 and all(len(line) <= MAX_LINE for line in lines)
        text = " ".join(lines)
        text = re.sub(r"^Speaker \d: ", "", text)
        words += text.split()
    assert words == _all_words()  # every word, in order, nothing added
    assert cues[0][2][0].startswith("Speaker 1: ") and "Speaker 2: " in srt
    assert sum("Speaker 1: " in b for b in blocks) == 2  # only where a turn starts
    starts = [c[0] for c in cues]
    assert starts == sorted(starts)


def test_vtt_escapes_markup_and_marks_voices(service):
    vtt = _export(service, "vtt")
    assert vtt.startswith("WEBVTT\n")
    assert "&lt;b&gt;tags&lt;/b&gt; &amp; ampersands" in vtt.replace("\n", " ")
    assert "<b>" not in vtt
    cues = _cues(vtt, ".")
    assert all(c[2][0].startswith("<v Speaker ") and c[2][-1].endswith("</v>") for c in cues)
    plain = [re.sub(r"</?v[^>]*>", "", " ".join(c[2])) for c in cues]
    words = " ".join(plain).replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&").split()
    assert words == _all_words()


def test_json_has_everything_the_specification_lists(service):
    call = service.methods()
    segment = repo.get_transcript(service.db, service.project_id)["segments"][0]
    call["transcript.editText"]({"project_id": service.project_id, "segment_id": segment["id"],
                                 "text": "Hello, and welcome to the show."})
    call["transcript.renameSpeaker"]({"project_id": service.project_id,
                                      "speaker_id": segment["speaker_id"], "name": "Ada"})
    data = json.loads(_export(service, "json"))
    assert data["format"] == "verbatim-transcript" and data["version"] == 1
    assert data["project"]["title"] == "Rivers & Bridges" and data["project"]["source"]["name"] == "jfk.wav"
    assert "/" not in data["project"]["source"]["name"]  # no local folder structure
    assert {s["display_name"] for s in data["speakers"]} == {"Ada", "Speaker 2"}
    first = data["segments"][0]
    assert first["speaker"] == "Ada" and first["speaker_id"] == segment["speaker_id"]
    assert first["text"] == "Hello, and welcome to the show." and first["text_edited"]
    assert first["original_text"].startswith("Hello and welcome to the show. Today")  # the AI's text
    assert all(w["start"] is not None and w["end"] is not None for s in data["segments"] for w in s["words"])
    assert data["segments"][1]["flags"][0]["type"] == "low_confidence"
    assert data["segments"][1]["review_priority"] == "medium"
    original = data["original"]["segments"]
    assert [s["text"] for s in original] == [t for _, t in TEXTS]  # the AI transcript as first stored
    assert len(original[0]["words"]) == len(TEXTS[0][1].split())


def test_csv_rows_are_safe_for_spreadsheets(service):
    content = _export(service, "csv")
    assert content.startswith("﻿")
    rows = list(csv.reader(io.StringIO(content.lstrip("﻿"))))
    assert rows[0][:6] == ["segment_id", "start", "end", "start_clock", "speaker", "text"]
    assert len(rows) == 1 + len(TEXTS)
    assert rows[2][5].startswith("'=SUM(A1)")  # a formula-like cell is defused
    assert rows[2][8] == "medium" and rows[2][9] == "low_confidence"
    words = list(csv.reader(io.StringIO(_export(service, "csv", words=True, bom=False))))
    assert words[0][5] == "word" and len(words) == 1 + len(_all_words())


def test_markdown_lists_speaker_turns_and_passages_to_review(service):
    md = _export(service, "md")
    assert md.startswith("# Rivers & Bridges\n")
    assert "**[00:00:00] Speaker 1**" in md and "Speaker 2**" in md
    assert "\\<b\\>tags\\</b\\>" in md  # markup is shown, not rendered
    review = md.split("## Passages to review", 1)[1]
    assert "low_confidence" in review and "SUM(A1)" in review
    assert "## Passages to review" not in _export(service, "md", review=False)


def test_unknown_format_is_refused(service):
    from verbatim_engine.errors import InvalidParams

    with pytest.raises(InvalidParams):
        service.export_render({"project_id": service.project_id, "format": "docx"})
    assert set(service.methods()["export.formats"]({})["formats"]) == {"txt", "srt", "vtt", "json", "csv", "md"}


def _plain(segments):
    return {"segments": segments, "speakers": [], "merged_speakers": []}


def test_cues_without_word_times_and_short_words():
    from verbatim_engine.export.subtitles import MIN_SECONDS, cues

    no_times = {"id": 1, "start": 10.0, "end": 14.0, "text": "one two three four", "speaker_id": None,
                "words": [{"start": None, "end": None, "text": " one"}]}
    short = {"id": 2, "start": 20.0, "end": 20.2, "text": "Yes.", "speaker_id": None,
             "words": [{"start": 20.0, "end": 20.2, "text": " Yes."}]}
    following = {"id": 3, "start": 20.5, "end": 21.0, "text": "No.", "speaker_id": None,
                 "words": [{"start": 20.5, "end": 21.0, "text": " No."}]}
    out = cues(_plain([no_times, short, following]))
    assert out[0].text == "one two three four" and (out[0].start, out[0].end) == (10.0, 14.0)
    assert out[1].end == pytest.approx(20.5)  # held, but not into the next cue
    assert out[2].end - out[2].start >= MIN_SECONDS - 1e-9
