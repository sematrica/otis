"""Full-text transcript search (SQLite FTS5, schema version 4).

The user's text is never passed to FTS5 as query syntax: it is split into
words and "quoted phrases", each quoted for FTS5, and the words are matched
as prefixes (so "brid" finds "bridge"); all of them must occur in the
passage. Case and diacritics are ignored. Languages written without spaces
between words (Chinese, Japanese, Thai, ...) have no word boundaries for the
index to use; for them the passages are searched by substring instead.

Results are passages (segments) with the matched words marked as character
ranges of the passage text, the project, the time and the speaker.
"""

from __future__ import annotations

import re
from typing import Any

from .errors import InvalidParams
from .storage.db import Database

MAX_LIMIT = 500
MAX_QUERY_CHARS = 500
_TERMS = re.compile(r'"([^"]*)"|(\S+)')
#: scripts without spaces between words (CJK, kana, Thai, Lao, Khmer, Myanmar)
_UNSPACED = re.compile(r"[฀-໿က-႟ក-៿぀-ヿ㐀-䶿一-鿿"
                       r"가-힯豈-﫿ｦ-ﾟ]")
_START, _END = "\x02", "\x03"


def parse_terms(text: str) -> list[tuple[str, bool]]:
    """``[(term, is_phrase)]`` from the user's text."""
    out = []
    for phrase, word in _TERMS.findall(text[:MAX_QUERY_CHARS]):
        term = (phrase if phrase else word).strip()
        # characters FTS5 treats as separators carry no meaning on their own
        if term and re.search(r"\w", term):
            out.append((term, bool(phrase)))
    return out


def fts_query(terms: list[tuple[str, bool]], include_original: bool) -> str:
    parts = []
    for term, phrase in terms:
        quoted = '"' + term.replace('"', '""') + '"'
        parts.append(quoted if phrase else quoted + "*")
    expression = " ".join(parts)
    return expression if include_original else f"text : ({expression})"


def _marked_to_ranges(marked: str) -> tuple[str, list[list[int]]]:
    plain, ranges, start = [], [], None
    position = 0
    for ch in marked:
        if ch == _START:
            start = position
        elif ch == _END:
            if start is not None and position > start:
                ranges.append([start, position])
            start = None
        else:
            plain.append(ch)
            position += 1
    return "".join(plain), ranges


def _substring_ranges(text: str, terms: list[tuple[str, bool]]) -> list[list[int]]:
    folded = text.casefold()
    ranges = []
    for term, _ in terms:
        needle = term.casefold()
        start = folded.find(needle)
        while needle and start >= 0:
            ranges.append([start, start + len(needle)])
            start = folded.find(needle, start + len(needle))
    return sorted(ranges)


def search(db: Database, text: str, *, project_id: str | None = None, speaker_id: int | None = None,
           include_original: bool = False, limit: int = 100) -> dict[str, Any]:
    terms = parse_terms(text)
    if not terms:
        raise InvalidParams("Type a word or a \"phrase\" to search for.")
    limit = max(1, min(int(limit), MAX_LIMIT))
    filters, params = [], []
    if project_id is not None:
        filters.append("s.project_id = ?")
        params.append(project_id)
    if speaker_id is not None:
        filters.append("s.speaker_id = ?")
        params.append(int(speaker_id))
    unspaced = any(_UNSPACED.search(term) for term, _ in terms)
    columns = """s.id, s.project_id, s.start, s.end, s.speaker_id, s.text, s.original_text, s.text_edited,
                 p.title AS project_title, COALESCE(sp.name, sp.label) AS speaker"""
    joins = """JOIN projects p ON p.id = s.project_id
               LEFT JOIN speakers sp ON sp.id = s.speaker_id"""
    order = "s.start" if project_id is not None else "p.updated_at DESC, s.start"
    if unspaced:
        likes = []
        for term, _ in terms:
            pattern = "%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            column = "(s.text || ' ' || s.original_text)" if include_original else "s.text"
            likes.append(f"{column} LIKE ? ESCAPE '\\'")
            params.append(pattern)
        where = " AND ".join(likes + filters)
        rows = db.query(f"SELECT {columns} FROM segments s {joins} WHERE {where} ORDER BY {order} LIMIT ?",
                        [*params, limit + 1])
        results = [_result(r, r["text"], _substring_ranges(r["text"], terms)) for r in rows[:limit]]
        method = "substring"
    else:
        match = fts_query(terms, include_original)
        where = " AND ".join(["segments_fts MATCH ?", *filters])
        if project_id is None:
            order = "bm25(segments_fts), p.updated_at DESC, s.start"
        rows = db.query(
            f"""SELECT {columns}, highlight(segments_fts, 0, char(2), char(3)) AS marked
                FROM segments_fts JOIN segments s ON s.id = segments_fts.rowid {joins}
                WHERE {where} ORDER BY {order} LIMIT ?""",
            [match, *params, limit + 1],
        )
        results = []
        for r in rows[:limit]:
            plain, ranges = _marked_to_ranges(r["marked"])
            results.append(_result(r, plain, ranges))
        method = "index"
    return {"query": text, "terms": [t for t, _ in terms], "method": method, "results": results,
            "truncated": len(rows) > limit}


def _result(r, text: str, ranges: list[list[int]]) -> dict[str, Any]:
    return {
        "project_id": r["project_id"],
        "project_title": r["project_title"],
        "segment_id": r["id"],
        "start": r["start"],
        "end": r["end"],
        "speaker_id": r["speaker_id"],
        "speaker": r["speaker"],
        "text": text,
        "highlights": ranges,
        # the match is only in the AI's original wording (the user corrected it)
        "in_original_only": not ranges and bool(r["text_edited"]),
    }
