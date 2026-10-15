"""Logging setup. Logs go to stderr and a rotating file, never to stdout
(stdout carries the protocol)."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path


def configure(log_dir: Path | None, name: str, level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    fmt = logging.Formatter(f"%(asctime)s {name}[%(process)d] %(levelname)s %(name)s: %(message)s")
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setFormatter(fmt)
    root.addHandler(stderr)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "engine.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    # Third-party libraries are chatty at INFO.
    for noisy in ("faster_whisper", "httpx", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
