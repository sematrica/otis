"""Export Manager: plain text (with speaker turns when speakers are known),
SRT and WebVTT subtitles, JSON (everything, for reuse and audit), CSV and
Markdown."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable

from ..errors import InvalidParams
from .structured import render_csv, render_json, render_markdown
from .subtitles import render_srt, render_vtt
from .txt import render_txt

Renderer = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], str]

#: format -> (file extension, renderer)
FORMATS: dict[str, tuple[str, Renderer]] = {
    "txt": ("txt", render_txt),
    "srt": ("srt", render_srt),
    "vtt": ("vtt", render_vtt),
    "json": ("json", render_json),
    "csv": ("csv", render_csv),
    "md": ("md", render_markdown),
}
#: formats that need word times
WORD_FORMATS = {"srt", "vtt", "json", "csv"}


def render(fmt: str, project: dict[str, Any], transcript: dict[str, Any], options: dict[str, Any] | None = None) -> str:
    if fmt not in FORMATS:
        raise InvalidParams(f"Unsupported export format '{fmt}'.", data={"formats": list(FORMATS)})
    return FORMATS[fmt][1](project, transcript, options or {})


def suggested_filename(project: dict[str, Any], fmt: str) -> str:
    stem = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "_", project["title"]).strip(" .") or "transcript"
    return f"{stem}.{FORMATS[fmt][0]}"


def write_atomic(path: str, content: str) -> int:
    target = Path(path).expanduser()
    if not target.is_absolute():
        raise InvalidParams("The export path must be absolute.")
    if not target.parent.is_dir():
        raise InvalidParams("The export folder does not exist.")
    data = content.encode("utf-8")
    fd, tmp = tempfile.mkstemp(prefix=".verbatim-export-", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return len(data)
