"""Optional transcript review by a language model.

The model reads the transcript text (never the audio) in batches and may
only *suggest*:

* ``suspicious`` - a passage that reads like a recognition error;
* ``punctuation`` - the same words with better punctuation or capitals;
* ``terminology`` - one spelling of a name or term that is inconsistent with
  how it is spelled elsewhere in the transcript;
* ``retranscribe`` - a passage worth decoding again from the audio.

Every suggestion is checked before it is stored: a punctuation suggestion
that changes any word is rejected (``invalid``, with the reason), a
terminology suggestion must name a spelling that is actually in the passage,
and every suggestion must refer to a passage that was sent. Nothing is
applied automatically; the user accepts or rejects each one, and accepting
goes through the transcript editor (undoable, in the edit history). The
model's guesses about plausible wording are never taken as evidence of what
was said: a suspicious or re-transcribe suggestion at most marks a passage
for review or for an acoustic re-check.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from ..asr.merge import normalize_token

KINDS = ("suspicious", "punctuation", "terminology", "retranscribe")
#: Words per request (keeps prompts well inside small local models' context).
BATCH_WORDS = 900
BATCH_SEGMENTS = 80
MAX_PER_SEGMENT = 4

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment_id": {"type": "integer"},
                    "kind": {"type": "string", "enum": list(KINDS)},
                    "text": {"type": "string"},
                    "term": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["segment_id", "kind", "text", "term", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["suggestions"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You help a person check an automatic speech-recognition transcript against the original \
recording. You only see the text; you cannot hear the audio, so you must never decide what was said.

Each line of the transcript has the form [ID] Speaker: text. Report problems as suggestions:

- "suspicious": the passage reads like a recognition error (words that make no sense in context, \
garbled phrases, sign-off phrases that look out of place). Put an empty string in "text" and "term".
- "punctuation": the passage needs different punctuation or capital letters. Put the complete passage \
with corrected punctuation and capitals in "text". Keep every word exactly as it is: do not add, remove, \
reorder or change any word. Put an empty string in "term".
- "terminology": a name or technical term is spelled differently here than elsewhere in the transcript. \
Put the spelling used in this passage in "term" and the spelling to use instead in "text".
- "retranscribe": the passage should be decoded again from the audio (likely misheard). Put an empty \
string in "text" and "term".

Always explain briefly in "reason". Never rewrite or paraphrase what people said. If nothing needs \
attention, return an empty list. Answer with JSON only: {"suggestions": [{"segment_id", "kind", "text", \
"term", "reason"}]}."""


def _words(text: str) -> list[str]:
    return [t for t in (normalize_token(w) for w in re.split(r"[\s\-‐‑‒–—]+", text)) if t]


def batches(segments: Sequence[dict[str, Any]], speakers: dict[int, str]) -> list[list[dict[str, Any]]]:
    out: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    words = 0
    for seg in segments:
        n = len(seg["text"].split())
        if current and (words + n > BATCH_WORDS or len(current) >= BATCH_SEGMENTS):
            out.append(current)
            current, words = [], 0
        current.append(seg)
        words += n
    if current:
        out.append(current)
    return out


def prompt_for(batch: Sequence[dict[str, Any]], speakers: dict[int, str]) -> str:
    lines = []
    for seg in batch:
        who = speakers.get(seg.get("speaker_id"), "Unknown speaker")
        lines.append(f"[{seg['id']}] {who}: {seg['text'].strip()}")
    return "Transcript:\n" + "\n".join(lines)


def validate(item: Any, by_id: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
    """Turn one raw suggestion into a stored record (status ``pending`` or
    ``invalid`` with the reason), or None if it is empty or a no-op."""
    if not isinstance(item, dict):
        return None
    kind = item.get("kind")
    reason = str(item.get("reason") or "").strip()[:500]
    text = str(item.get("text") or "").strip()
    term = str(item.get("term") or "").strip()
    try:
        segment_id = int(item.get("segment_id"))
    except (TypeError, ValueError):
        segment_id = None
    record = {"segment_id": segment_id, "kind": kind if kind in KINDS else "suspicious", "original_text": "",
              "suggested_text": None, "reason": reason, "status": "pending", "validation": None}
    seg = by_id.get(segment_id) if segment_id is not None else None
    if kind not in KINDS:
        return {**record, "status": "invalid", "validation": f"unknown kind {kind!r}"}
    if seg is None:
        return {**record, "segment_id": None, "status": "invalid", "validation": "refers to a passage that was not sent"}
    original = seg["text"].strip()
    record["original_text"] = original
    if kind == "punctuation":
        if not text or text == original:
            return None
        if _words(text) != _words(original):
            return {**record, "suggested_text": text, "status": "invalid",
                    "validation": "changes words, not only punctuation"}
        record["suggested_text"] = text
    elif kind == "terminology":
        if not term or not text or _words(term) == _words(text):
            return None
        pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
        if not pattern.search(original):
            return {**record, "status": "invalid", "validation": f"'{term}' is not in this passage"}
        record["suggested_text"] = pattern.sub(text, original)
        record["reason"] = reason or f"'{term}' is spelled '{text}' elsewhere"
    if not record["reason"] and kind in ("suspicious", "retranscribe"):
        record["reason"] = "no reason given"
    return record


def collect(parsed: dict[str, Any], batch: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {seg["id"]: seg for seg in batch}
    items = parsed.get("suggestions")
    if not isinstance(items, list):
        return [{"segment_id": None, "kind": "suspicious", "original_text": "", "suggested_text": None, "reason": "",
                 "status": "invalid", "validation": "the answer has no suggestion list"}]
    out: list[dict[str, Any]] = []
    per_segment: dict[Any, int] = {}
    seen: set[tuple] = set()
    for item in items:
        record = validate(item, by_id)
        if record is None:
            continue
        key = (record["segment_id"], record["kind"], record["suggested_text"])
        if key in seen:
            continue
        seen.add(key)
        per_segment[record["segment_id"]] = per_segment.get(record["segment_id"], 0) + 1
        if per_segment[record["segment_id"]] > MAX_PER_SEGMENT:
            continue
        out.append(record)
    return out
