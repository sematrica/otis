"""Helpers shared by the export formats."""

from __future__ import annotations

from typing import Any

from ..verify.review import flag_severity


def speaker_names(transcript: dict[str, Any]) -> dict[int, str]:
    """Display name of every speaker id, merged speakers included."""
    names = {sp["id"]: sp["display_name"] for sp in transcript.get("speakers") or []}
    names.update({sp["id"]: sp["display_name"] for sp in transcript.get("merged_speakers") or []})
    return names


def spoken_segments(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    return [seg for seg in transcript["segments"] if seg["text"].strip()]


def open_doubts(segment: dict[str, Any]) -> list[str]:
    """Types of the flags that still ask for review (the review priority's sources)."""
    if any(f.get("type") == "reviewed" for f in segment["flags"]):
        return [f["type"] for f in segment["flags"] if f.get("type") == "marked_uncertain"]
    return [f["type"] for f in segment["flags"] if flag_severity(f) is not None]


def source_name(project: dict[str, Any]) -> str:
    """The source's file name or URL (never the full local path)."""
    source = project["source"]
    if source.get("url"):
        return source["url"]
    path = source.get("path") or ""
    return path.replace("\\", "/").rsplit("/", 1)[-1]
