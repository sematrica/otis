"""Export Manager. Plain text (with speaker turns when speakers are known);
SRT, VTT, JSON, CSV and Markdown arrive in Stage 4."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable

from ..errors import InvalidParams
from .txt import render_txt

Renderer = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], str]

FORMATS: dict[str, tuple[str, Renderer]] = {
    "txt": ("txt", render_txt),
}


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
