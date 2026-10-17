"""JSON, CSV and Markdown exports."""

from __future__ import annotations

import csv
import io
from typing import Any

from ..util import jsonsafe
from ..util.ids import utc_now
from ..util.timecode import format_clock
from .common import open_doubts, source_name, speaker_names, spoken_segments

JSON_FORMAT = "verbatim-transcript"
JSON_VERSION = 1


def _project_meta(project: dict[str, Any], transcript: dict[str, Any]) -> dict[str, Any]:
    effective = transcript.get("effective") or {}
    model = effective.get("model") or {}
    return {
        "id": project["id"],
        "title": project["title"],
        "source": {"kind": project["source"]["kind"], "name": source_name(project),
                   "url": project["source"].get("url"), "size_bytes": project["source"].get("size_bytes")},
        "duration": project.get("duration"),
        "language": transcript.get("language"),
        "created_at": project["created_at"],
        "updated_at": project["updated_at"],
        "transcription": {
            "job_id": transcript.get("job_id"),
            "model": model.get("display_name"),
            "model_id": model.get("id"),
            "backend": effective.get("backend"),
            "mode": (transcript.get("_job") or {}).get("mode"),
            "validation": effective.get("validation"),
        },
        "revision": transcript.get("revision"),
    }


def _speakers(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"id": sp["id"], "label": sp["label"], "name": sp["name"], "display_name": sp["display_name"],
             "origin": sp["origin"], "merged_into": sp["merged_into"]}
            for sp in [*(transcript.get("speakers") or []), *(transcript.get("merged_speakers") or [])]]


def _words(seg: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"start": w["start"], "end": w["end"], "text": w["text"].strip(), "probability": w["probability"],
             "timing_source": w["timing_source"], "speaker_id": w["speaker_id"]} for w in seg.get("words") or []]


def render_json(project: dict[str, Any], transcript: dict[str, Any], options: dict[str, Any]) -> str:
    """Everything needed to reuse or audit the transcript: metadata, speakers,
    segments with words, corrections next to the AI text, every flag, and the
    AI transcript as first stored (``original``)."""
    names = speaker_names(transcript)
    segments = []
    for seg in transcript["segments"]:
        segments.append({
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "speaker_id": seg.get("speaker_id"),
            "speaker": names.get(seg.get("speaker_id")),
            "original_speaker_id": seg.get("original_speaker_id"),
            "text": seg["text"].strip(),
            "original_text": seg["original_text"].strip(),
            "text_edited": seg["text_edited"],
            "speaker_edited": seg["speaker_edited"],
            "language": seg.get("language"),
            "review_priority": seg.get("review_priority"),
            "flags": seg["flags"],
            "words": _words(seg),
        })
    original = options.get("_original")
    payload = {
        "format": JSON_FORMAT,
        "version": JSON_VERSION,
        "exported_at": utc_now(),
        "project": _project_meta(project, transcript),
        "speakers": _speakers(transcript),
        "segments": segments,
        "original": None if original is None else {
            "job_id": original.get("job_id"),
            "created_at": original.get("created_at"),
            "speakers": [{"id": sp["id"], "label": sp["label"]} for sp in original.get("speakers") or []],
            "segments": [{"id": s["id"], "start": s["start"], "end": s["end"], "speaker_id": s.get("speaker_id"),
                          "text": s["text"].strip(), "flags": s["flags"], "words": _words(s)}
                         for s in original.get("segments") or []],
        },
    }
    return jsonsafe.dumps(payload, ensure_ascii=False, indent=2) + "\n"


#: Spreadsheet programs run cells that start like a formula; such cells are
#: prefixed with an apostrophe (OWASP's advice against CSV injection).
_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def _cell(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(_FORMULA_START):
        return "'" + value
    return value


def render_csv(project: dict[str, Any], transcript: dict[str, Any], options: dict[str, Any]) -> str:
    """One row per segment, or per word with ``{"words": true}``. UTF-8 with a
    byte-order mark so spreadsheet programs read non-ASCII text correctly."""
    names = speaker_names(transcript)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\r\n")
    if options.get("words"):
        writer.writerow(["segment_id", "word_index", "start", "end", "speaker", "word", "probability",
                         "timing_source"])
        for seg in spoken_segments(transcript):
            for i, w in enumerate(seg.get("words") or []):
                writer.writerow([seg["id"], i, w["start"], w["end"], _cell(names.get(w["speaker_id"], "")),
                                 _cell(w["text"].strip()), w["probability"], w["timing_source"]])
    else:
        writer.writerow(["segment_id", "start", "end", "start_clock", "speaker", "text", "original_text",
                         "text_edited", "review_priority", "open_flags"])
        for seg in spoken_segments(transcript):
            writer.writerow([seg["id"], round(seg["start"], 3), round(seg["end"], 3), format_clock(seg["start"]),
                             _cell(names.get(seg.get("speaker_id"), "")), _cell(seg["text"].strip()),
                             _cell(seg["original_text"].strip()), int(seg["text_edited"]),
                             seg.get("review_priority") or "", ";".join(open_doubts(seg))])
    content = buffer.getvalue()
    return ("﻿" + content) if options.get("bom", True) else content


def _md_escape(text: str) -> str:
    out = text
    for ch in ("\\", "`", "*", "_", "[", "]", "<", ">", "|"):
        out = out.replace(ch, "\\" + ch)
    return out


def render_markdown(project: dict[str, Any], transcript: dict[str, Any], options: dict[str, Any]) -> str:
    names = speaker_names(transcript)
    meta = _project_meta(project, transcript)
    lines = [f"# {_md_escape(project['title'])}", ""]
    lines.append(f"- Source: {_md_escape(source_name(project))}")
    if project.get("duration") is not None:
        lines.append(f"- Duration: {format_clock(project['duration'])}")
    if transcript.get("language"):
        lines.append(f"- Language: {transcript['language']}")
    if meta["transcription"]["model"]:
        lines.append(f"- Transcribed with: {_md_escape(meta['transcription']['model'])}")
    lines.append("")
    review: list[str] = []
    previous = object()
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            lines.append(" ".join(paragraph))
            lines.append("")
            paragraph.clear()

    for seg in spoken_segments(transcript):
        speaker = seg.get("speaker_id")
        if speaker != previous or not names:
            flush()
            label = names.get(speaker)
            head = f"**[{format_clock(seg['start'])}]{' ' + _md_escape(label) if label else ''}**"
            paragraph.append(head)
            previous = speaker
        paragraph.append(_md_escape(seg["text"].strip()))
        doubts = open_doubts(seg)
        if doubts and seg.get("review_priority") in ("high", "medium"):
            review.append(f"- [{format_clock(seg['start'])}] {', '.join(sorted(set(doubts)))}: "
                          f"“{_md_escape(seg['text'].strip())}”")
    flush()
    if review and options.get("review", True):
        lines += ["## Passages to review", "", *review, ""]
    return "\n".join(lines).rstrip("\n") + "\n"
