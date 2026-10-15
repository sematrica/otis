"""Timestamp formatting."""

from __future__ import annotations

import math


def _split(seconds: float) -> tuple[int, int, int, int]:
    if seconds is None or math.isnan(seconds) or seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return hours, minutes, secs, ms


def format_clock(seconds: float) -> str:
    """``HH:MM:SS`` (seconds truncated, not rounded up, so a segment starting at
    59.9 s is labelled 00:00:59 rather than appearing to start a second late)."""
    if seconds is None or math.isnan(seconds) or seconds < 0:
        seconds = 0.0
    whole = int(math.floor(seconds))
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_precise(seconds: float, separator: str = ".") -> str:
    """``HH:MM:SS.mmm`` (or ``,`` separator for SRT)."""
    hours, minutes, secs, ms = _split(seconds)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{ms:03d}"


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    return format_clock(seconds)
