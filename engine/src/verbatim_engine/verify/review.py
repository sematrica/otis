"""Review priority of a segment, from its flags.

Every flag that asks for a look has a severity: Stage 3 flags carry their own
(``"severity"``), earlier flags get the defaults below. A segment's priority
is its most severe flag; segments without such flags need no review.

The defaults reflect what was measured (see the Stage 2 and 3 reports): an
uncertain speaker *change* fired on most correct changes and never on a wrong
word, so it is low; a user's own "uncertain" mark is high.
"""

from __future__ import annotations

from typing import Any, Iterable

LEVELS = {"low": 1, "medium": 2, "high": 3}

DEFAULT_SEVERITY = {
    "marked_uncertain": "high",
    "alignment_failed": "medium",
    "overlapping_speech": "medium",
    "speaker_uncertain": "medium",
    "speaker_unassigned": "medium",
    "speaker_change_uncertain": "low",
    "boundary_duplicate_removed": "low",
}

#: Flags that only record what was done (no review needed).
INFORMATIONAL = {"split_at_speaker_change", "chunk_boundary", "trimmed_at_chunk_boundary", "rechecked", "reviewed"}


def flag_severity(flag: dict[str, Any]) -> str | None:
    kind = flag.get("type")
    if kind in INFORMATIONAL or flag.get("resolved"):
        return None
    severity = flag.get("severity") or DEFAULT_SEVERITY.get(kind)
    return severity if severity in LEVELS else None


def review_priority(flags: Iterable[dict[str, Any]]) -> str | None:
    """Most severe open flag; None once the user has marked the passage reviewed
    (a user's own "uncertain" mark still counts)."""
    flags = list(flags)
    reviewed = any(f.get("type") == "reviewed" for f in flags)
    best = None
    for flag in flags:
        if reviewed and flag.get("type") != "marked_uncertain":
            continue
        severity = flag_severity(flag)
        if severity is not None and (best is None or LEVELS[severity] > LEVELS[best]):
            best = severity
    return best
