"""Strict JSON for everything that leaves the engine.

Python's ``json`` writes NaN and infinity as bare ``NaN``/``Infinity`` and a
lone UTF-16 surrogate as ``\\udXXX``: neither is JSON, and a strict reader
(the app's) rejects the whole message. Values like these are replaced (a
non-finite number by ``null``, a lone surrogate by U+FFFD) and reported, so
one odd value can never make a reply unreadable.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

#: a lone surrogate escape, as ``json.dumps(ensure_ascii=True)`` writes it
_LONE_SURROGATE = re.compile(r"\\ud[89ab][0-9a-f]{2}(?!\\ud[c-f][0-9a-f]{2})|(?<!\\ud[89ab][0-9a-f]{2})\\ud[c-f][0-9a-f]{2}")


def _clean_text(text: str) -> str:
    return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


def clean(value: Any, path: str = "$", found: list[str] | None = None) -> Any:
    """``value`` with non-finite numbers as None, lone surrogates as U+FFFD
    and keys JSON cannot hold as strings; the paths changed go to ``found``."""
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if found is not None:
            found.append(path)
        return None
    if isinstance(value, str):
        fixed = _clean_text(value)
        if fixed != value and found is not None:
            found.append(path)
        return fixed
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)) and key is not None:
                key = str(key)
            elif isinstance(key, str):
                key = _clean_text(key)
            out[key] = clean(item, f"{path}.{key}", found)
        return out
    if isinstance(value, (list, tuple)):
        return [clean(item, f"{path}[{i}]", found) for i, item in enumerate(value)]
    return value


def dumps(value: Any, *, found: list[str] | None = None, **options: Any) -> str:
    """``json.dumps`` that always produces JSON. Values it had to replace are
    listed in ``found`` (paths like ``$.segments[3].avg_logprob``)."""
    options.setdefault("default", str)
    try:
        text = json.dumps(value, allow_nan=False, **options)
        if options.get("ensure_ascii", True):
            valid = "\\ud" not in text or not _LONE_SURROGATE.search(text)
        else:
            text.encode("utf-8")  # a lone surrogate cannot be encoded
            valid = True
    except (TypeError, ValueError):  # non-finite numbers, odd keys; UnicodeEncodeError is a ValueError
        valid = False
    if valid:
        return text
    replaced: list[str] = []
    text = json.dumps(clean(value, found=replaced), allow_nan=False, **options)
    if found is not None:
        found.extend(replaced)
    return text
