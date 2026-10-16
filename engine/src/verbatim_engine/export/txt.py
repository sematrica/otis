"""Plain-text export."""

from __future__ import annotations

import os
from typing import Any

from ..util.timecode import format_clock, format_duration


def render_txt(project: dict[str, Any], transcript: dict[str, Any], options: dict[str, Any]) -> str:
    include_timestamps = options.get("timestamps", True)
    include_header = options.get("header", True)
    lines: list[str] = []
    if include_header:
        effective = transcript.get("effective") or {}
        model = (effective.get("model") or {}).get("display_name")
        backend = effective.get("backend")
        lines.append(project["title"])
        lines.append(f"Source: {os.path.basename(project['source'].get('path') or '') or project['source'].get('url') or ''}")
        lines.append(f"Duration: {format_duration(project.get('duration'))}")
        if transcript.get("language"):
            lines.append(f"Language: {transcript['language']}")
        if model:
            lines.append(f"Transcribed with: {model}" + (f" ({backend})" if backend else ""))
        lines.append("")
    names = {sp["id"]: sp["display_name"] for sp in transcript.get("speakers") or []}
    names.update({sp["id"]: sp["display_name"] for sp in transcript.get("merged_speakers") or []})
    if names and options.get("speakers", True):
        # One block per speaker turn, as in "[00:01:12] Speaker 1" + text.
        previous = object()
        block: list[str] = []
        for seg in transcript["segments"]:
            text = seg["text"].strip()
            if not text:
                continue
            speaker = seg.get("speaker_id")
            if speaker != previous:
                if block:
                    lines.append(" ".join(block))
                    lines.append("")
                    block = []
                label = names.get(speaker, "Unknown speaker")
                lines.append(f"[{format_clock(seg['start'])}] {label}" if include_timestamps else label)
                previous = speaker
            block.append(text)
        if block:
            lines.append(" ".join(block))
    else:
        for seg in transcript["segments"]:
            text = seg["text"].strip()
            if not text:
                continue
            lines.append(f"[{format_clock(seg['start'])}] {text}" if include_timestamps else text)
    return "\n".join(lines).rstrip("\n") + "\n"
