"""User corrections to a transcript, with undo and redo.

Every correction runs in one database transaction that also records an
``edits`` row holding the complete state, before and after, of every speaker
and segment (with its words) the correction touched. Undo writes the
"before" state back, redo the "after" state; edits are undone in reverse
order and a new correction discards anything that was undone.

Corrections never touch the AI-generated original: it is stored separately
(``original_transcripts``), each segment keeps ``original_text`` and
``original_speaker_id``, and each word its ``original_text``.

Words always match the current text: editing a segment's text re-derives its
words, keeping the timing (and AI text) of every word that is still there and
interpolating times for new words (``timing_source = "interpolated"``).
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any, Callable, Iterable

from ..asr.merge import normalize_token
from ..errors import InvalidJobState, InvalidParams, NotFound
from ..storage.db import Database
from ..util.ids import utc_now
from ..util.timecode import format_clock

MAX_NAME_LENGTH = 100
MAX_TEXT_LENGTH = 20_000
#: Flags that record alternative wordings found by re-checking.
WORDING_FLAGS = ("retranscribed", "alternatives", "recovered_speech")

_SPEAKER_COLUMNS = ("id", "project_id", "label", "name", "origin", "cluster_index", "color_index", "merged_into", "created_at")
_SEGMENT_COLUMNS = (
    "id", "project_id", "job_id", "ordinal", "start", "end", "original_text", "text", "language", "avg_logprob",
    "no_speech_prob", "compression_ratio", "temperature", "chunk_index", "speaker_id", "original_speaker_id",
    "speaker_confidence", "text_edited", "speaker_edited", "flags_json",
)
_WORD_COLUMNS = (
    "id", "segment_id", "ordinal", "start", "end", "text", "probability", "timing_source", "speaker_id",
    "whisper_start", "whisper_end", "alignment_score", "original_text",
)


class TranscriptEditor:
    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------ speakers

    def rename_speaker(self, project_id: str, speaker_id: int, name: str | None) -> str | None:
        name = (name or "").strip()[:MAX_NAME_LENGTH] or None
        with self.db.transaction() as conn:
            row = self._speaker(conn, project_id, speaker_id)
            if row["name"] == name:
                return None
            before_name = row["name"] or row["label"]
            after_name = name or row["label"]
            summary = f"Rename “{before_name}” to “{after_name}”"

            def mutate(c):
                c.execute("UPDATE speakers SET name = ? WHERE id = ?", (name, speaker_id))

            return self._record(conn, project_id, "rename_speaker", summary, {speaker_id}, set(), mutate)

    def create_speaker(self, project_id: str, name: str | None = None) -> tuple[str, int]:
        with self.db.transaction() as conn:
            self._ensure_editable(conn, project_id)
            created: list[int] = []

            def mutate(c):
                created.append(self._insert_speaker(c, project_id, name))

            summary = f"Add speaker “{(name or '').strip() or 'new speaker'}”"
            self._record(conn, project_id, "create_speaker", summary, set(), set(), mutate,
                         created_speakers=lambda: set(created))
            return summary, created[0]

    def merge_speakers(self, project_id: str, speaker_ids: Iterable[int], into: int) -> str:
        sources = {int(s) for s in speaker_ids} - {int(into)}
        if not sources:
            raise InvalidParams("Choose at least one other speaker to merge.")
        with self.db.transaction() as conn:
            target = self._speaker(conn, project_id, into)
            names = [self._display(self._speaker(conn, project_id, s)) for s in sorted(sources)]
            affected = self._segments_mentioning(conn, project_id, sources)

            def mutate(c):
                marks = ",".join("?" * len(sources))
                c.execute(
                    f"UPDATE segments SET speaker_id = ?, speaker_edited = 1 WHERE project_id = ? AND speaker_id IN ({marks})",
                    (into, project_id, *sources),
                )
                c.execute(
                    f"""UPDATE words SET speaker_id = ? WHERE speaker_id IN ({marks})
                        AND segment_id IN (SELECT id FROM segments WHERE project_id = ?)""",
                    (into, *sources, project_id),
                )
                for seg_id in affected:
                    row = c.execute("SELECT speaker_id, flags_json FROM segments WHERE id = ?", (seg_id,)).fetchone()
                    flags = json.loads(row["flags_json"])
                    for flag in flags:
                        if "other_speakers" in flag:
                            mapped = {into if o in sources else o for o in flag["other_speakers"]}
                            flag["other_speakers"] = sorted(mapped - {row["speaker_id"]})
                    c.execute("UPDATE segments SET flags_json = ? WHERE id = ?", (json.dumps(flags), seg_id))
                c.execute(f"UPDATE speakers SET merged_into = ? WHERE id IN ({marks})", (into, *sources))

            summary = f"Merge {', '.join(f'“{n}”' for n in names)} into “{self._display(target)}”"
            return self._record(conn, project_id, "merge_speakers", summary, sources | {into}, affected, mutate)

    # ------------------------------------------------------------------ segments

    def assign_speaker(self, project_id: str, segment_ids: Iterable[int], speaker_id: int | None = None,
                       new_speaker_name: str | None = None) -> tuple[str, int]:
        segment_ids = {int(s) for s in segment_ids}
        if not segment_ids:
            raise InvalidParams("Choose at least one segment.")
        if (speaker_id is None) == (new_speaker_name is None):
            raise InvalidParams("Give either speaker_id or new_speaker_name.")
        with self.db.transaction() as conn:
            rows = [self._segment(conn, project_id, s) for s in segment_ids]
            created: list[int] = []
            if speaker_id is not None:
                target = self._speaker(conn, project_id, speaker_id)
                if target["merged_into"] is not None:
                    raise InvalidParams("That speaker was merged into another one.")
                target_name = self._display(target)
            else:
                target_name = new_speaker_name.strip() or "new speaker"

            def mutate(c):
                sid = speaker_id
                if sid is None:
                    sid = self._insert_speaker(c, project_id, new_speaker_name)
                    created.append(sid)
                for row in rows:
                    c.execute(
                        "UPDATE segments SET speaker_id = ?, speaker_edited = ? WHERE id = ?",
                        (sid, int(row["original_speaker_id"] != sid), row["id"]),
                    )
                    c.execute("UPDATE words SET speaker_id = ? WHERE segment_id = ?", (sid, row["id"]))

            what = "segment" if len(rows) == 1 else f"{len(rows)} segments"
            summary = f"Assign {what} to “{target_name}”"
            speakers = {speaker_id} if speaker_id is not None else set()
            self._record(conn, project_id, "assign_speaker", summary, speakers, segment_ids, mutate,
                         created_speakers=lambda: set(created))
            return summary, (speaker_id if speaker_id is not None else created[0])

    def split_segment(self, project_id: str, segment_id: int, word_index: int) -> tuple[str, int]:
        with self.db.transaction() as conn:
            row = self._segment(conn, project_id, segment_id)
            words = self._words(conn, segment_id)
            if not 1 <= word_index < len(words):
                raise InvalidParams(f"word_index must be between 1 and {len(words) - 1} for this segment.")
            created: list[int] = []

            def mutate(c):
                first, second = words[:word_index], words[word_index:]
                split_at = second[0]["start"] if second[0]["start"] is not None else (row["start"] + row["end"]) / 2
                first_end = first[-1]["end"] if first[-1]["end"] is not None else split_at
                values = {k: row[k] for k in _SEGMENT_COLUMNS if k != "id"}
                part2_text = _join(w["text"] for w in second)
                part2_original = _join(w["original_text"] for w in second if w["original_text"] is not None)
                values.update(start=split_at, text=part2_text, original_text=part2_original,
                              text_edited=int(part2_text != part2_original))
                new_id = _insert(c, "segments", values)
                created.append(new_id)
                for ordinal, w in enumerate(second):
                    c.execute("UPDATE words SET segment_id = ?, ordinal = ? WHERE id = ?", (new_id, ordinal, w["id"]))
                part1_text = _join(w["text"] for w in first)
                part1_original = _join(w["original_text"] for w in first if w["original_text"] is not None)
                c.execute(
                    "UPDATE segments SET end = ?, text = ?, original_text = ?, text_edited = ? WHERE id = ?",
                    (max(row["start"], min(first_end, split_at)), part1_text, part1_original,
                     int(part1_text != part1_original), segment_id),
                )

            summary = f"Split segment at {format_clock(words[word_index]['start'] or row['start'])}"
            self._record(conn, project_id, "split_segment", summary, set(), {segment_id}, mutate,
                         created_segments=lambda: set(created))
            return summary, created[0]

    def merge_segments(self, project_id: str, segment_ids: Iterable[int]) -> str:
        wanted = {int(s) for s in segment_ids}
        if len(wanted) < 2:
            raise InvalidParams("Choose at least two adjacent segments to merge.")
        with self.db.transaction() as conn:
            for s in wanted:
                self._segment(conn, project_id, s)
            order = [r["id"] for r in conn.execute(
                "SELECT id FROM segments WHERE project_id = ? ORDER BY start, id", (project_id,))]
            positions = sorted(order.index(s) for s in wanted)
            if positions[-1] - positions[0] != len(positions) - 1:
                raise InvalidParams("Only adjacent segments can be merged.")
            ids = [order[p] for p in positions]
            rows = [self._segment(conn, project_id, s) for s in ids]

            def mutate(c):
                target = rows[0]
                all_words = [w for s in ids for w in self._words(c, s)]
                for ordinal, w in enumerate(all_words):
                    c.execute(
                        "UPDATE words SET segment_id = ?, ordinal = ?, speaker_id = ? WHERE id = ?",
                        (target["id"], ordinal, target["speaker_id"], w["id"]),
                    )
                flags: list[dict[str, Any]] = []
                for r in rows:
                    for flag in json.loads(r["flags_json"]):
                        if flag not in flags:
                            flags.append(flag)
                text = _join_segments(r["text"] for r in rows)
                original = _join_segments(r["original_text"] for r in rows)
                speakers_differ = any(r["speaker_id"] != target["speaker_id"] for r in rows)
                c.execute(
                    """UPDATE segments SET start = ?, end = ?, text = ?, original_text = ?, text_edited = ?,
                           speaker_edited = ?, flags_json = ? WHERE id = ?""",
                    (min(r["start"] for r in rows), max(r["end"] for r in rows), text, original,
                     int(any(r["text_edited"] for r in rows) or text != original),
                     int(speakers_differ or any(r["speaker_edited"] for r in rows)), json.dumps(flags), target["id"]),
                )
                for r in rows[1:]:
                    c.execute("DELETE FROM segments WHERE id = ?", (r["id"],))

            summary = f"Merge {len(ids)} segments at {format_clock(rows[0]['start'])}"
            return self._record(conn, project_id, "merge_segments", summary, set(), set(ids), mutate)

    def edit_text(self, project_id: str, segment_id: int, text: str, *, kind: str = "edit_text",
                  summary: str | None = None) -> str | None:
        if not isinstance(text, str) or not text.strip():
            raise InvalidParams("The text cannot be empty.")
        if len(text) > MAX_TEXT_LENGTH:
            raise InvalidParams("The text is too long.")
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
        with self.db.transaction() as conn:
            row = self._segment(conn, project_id, segment_id)
            if text == row["text"] or text.strip() == row["text"].strip():
                return None
            old_words = self._words(conn, segment_id)
            leading = " " if row["text"].startswith(" ") and not text.startswith(" ") else ""
            new_text = leading + text

            def mutate(c):
                derived = rederive_words(old_words, new_text, row["start"], row["end"])
                c.execute("DELETE FROM words WHERE segment_id = ?", (segment_id,))
                for ordinal, w in enumerate(derived):
                    _insert(c, "words", {**w, "segment_id": segment_id, "ordinal": ordinal,
                                         "speaker_id": row["speaker_id"]})
                c.execute(
                    "UPDATE segments SET text = ?, text_edited = ? WHERE id = ?",
                    (new_text, int(new_text.strip() != row["original_text"].strip()), segment_id),
                )

            summary = summary or f"Edit text at {format_clock(row['start'])}"
            return self._record(conn, project_id, kind, summary, set(), {segment_id}, mutate)

    def mark_uncertain(self, project_id: str, segment_id: int, uncertain: bool, note: str | None = None) -> str | None:
        with self.db.transaction() as conn:
            row = self._segment(conn, project_id, segment_id)
            flags = json.loads(row["flags_json"])
            marked = any(f.get("type") == "marked_uncertain" for f in flags)
            if marked == bool(uncertain) and not note:
                return None
            flags = [f for f in flags if f.get("type") != "marked_uncertain"]
            if uncertain:
                flags.append({"type": "marked_uncertain", "by": "user", "note": (note or "").strip()[:500] or None})

            def mutate(c):
                c.execute("UPDATE segments SET flags_json = ? WHERE id = ?", (json.dumps(flags), segment_id))

            summary = f"{'Mark' if uncertain else 'Unmark'} passage at {format_clock(row['start'])} as uncertain"
            return self._record(conn, project_id, "mark_uncertain", summary, set(), {segment_id}, mutate)

    def delete_segment(self, project_id: str, segment_id: int) -> str:
        """Remove a passage (for example, text with no acoustic support). Undoable."""
        with self.db.transaction() as conn:
            row = self._segment(conn, project_id, segment_id)

            def mutate(c):
                c.execute("DELETE FROM segments WHERE id = ?", (segment_id,))

            summary = f"Remove passage at {format_clock(row['start'])}"
            return self._record(conn, project_id, "delete_segment", summary, set(), {segment_id}, mutate)

    def choose_wording(self, project_id: str, segment_id: int, source: str) -> str | None:
        """Use a wording recorded by re-checking: ``first_pass`` (the text
        before re-transcription) or a candidate's ``source``."""
        with self.db.transaction() as conn:
            row = self._segment(conn, project_id, segment_id)
            flags = json.loads(row["flags_json"])
            text = None
            for flag in flags:
                if flag.get("type") not in WORDING_FLAGS:
                    continue
                if source == "first_pass" and flag.get("first_pass") is not None:
                    text = flag["first_pass"]
                for cand in flag.get("candidates", []):
                    if cand.get("source") == source and text is None:
                        text = cand.get("text")
            if text is None:
                raise InvalidParams(f"This passage has no '{source}' wording.", data={"source": source})
            text = " ".join(text.split())
            if not text:
                raise InvalidParams("That wording is empty. Remove the passage instead.")
            for flag in flags:
                if flag.get("type") in WORDING_FLAGS:
                    flag["resolved"] = True
                    flag["chosen_by_user"] = source
            if text == row["text"].strip():
                def mutate_flags(c):
                    c.execute("UPDATE segments SET flags_json = ? WHERE id = ?", (json.dumps(flags), segment_id))

                summary = f"Keep wording at {format_clock(row['start'])}"
                return self._record(conn, project_id, "choose_wording", summary, set(), {segment_id}, mutate_flags)
            old_words = self._words(conn, segment_id)
            new_text = (" " if row["text"].startswith(" ") else "") + text

            def mutate(c):
                derived = rederive_words(old_words, new_text, row["start"], row["end"])
                c.execute("DELETE FROM words WHERE segment_id = ?", (segment_id,))
                for ordinal, w in enumerate(derived):
                    _insert(c, "words", {**w, "segment_id": segment_id, "ordinal": ordinal,
                                         "speaker_id": row["speaker_id"]})
                c.execute(
                    "UPDATE segments SET text = ?, text_edited = ?, flags_json = ? WHERE id = ?",
                    (new_text, int(new_text.strip() != row["original_text"].strip()), json.dumps(flags), segment_id),
                )

            label = "first-pass" if source == "first_pass" else "alternative"
            summary = f"Use {label} wording at {format_clock(row['start'])}"
            return self._record(conn, project_id, "choose_wording", summary, set(), {segment_id}, mutate)

    def mark_reviewed(self, project_id: str, segment_ids: Iterable[int], reviewed: bool = True) -> str | None:
        """Record that the user checked these passages (their flags no longer
        ask for review), or undo that."""
        ids = sorted({int(i) for i in segment_ids})
        if not ids:
            raise InvalidParams("No segments given.")
        with self.db.transaction() as conn:
            rows = [self._segment(conn, project_id, sid) for sid in ids]
            changes = {}
            for row in rows:
                flags = json.loads(row["flags_json"])
                has = any(f.get("type") == "reviewed" for f in flags)
                if has == bool(reviewed):
                    continue
                flags = [f for f in flags if f.get("type") != "reviewed"]
                if reviewed:
                    flags.append({"type": "reviewed", "by": "user", "at": utc_now()})
                changes[row["id"]] = flags
            if not changes:
                return None

            def mutate(c):
                for sid, flags in changes.items():
                    c.execute("UPDATE segments SET flags_json = ? WHERE id = ?", (json.dumps(flags), sid))

            n = len(changes)
            what = f"{n} passage{'s' if n != 1 else ''}"
            summary = f"Mark {what} as reviewed" if reviewed else f"Mark {what} as needing review"
            return self._record(conn, project_id, "mark_reviewed", summary, set(), set(changes), mutate)

    def apply_recheck(self, project_id: str, job_id: str, updates: dict[int, dict[str, Any]], summary: str) -> str | None:
        """Write the result of an on-demand re-check job as one undoable edit.

        ``updates`` maps segment ids to ``{"text", "start", "end", "flags",
        "words" (None = unchanged)}``; words are dicts with the word columns."""
        if not updates:
            return None
        with self.db.transaction() as conn:
            rows = {sid: self._segment(conn, project_id, sid) for sid in updates}

            def mutate(c):
                for sid, update in updates.items():
                    row = rows[sid]
                    if update.get("words") is not None:
                        c.execute("DELETE FROM words WHERE segment_id = ?", (sid,))
                        for ordinal, w in enumerate(update["words"]):
                            _insert(c, "words", {**w, "segment_id": sid, "ordinal": ordinal,
                                                 "speaker_id": row["speaker_id"]})
                    c.execute(
                        "UPDATE segments SET text = ?, start = ?, end = ?, flags_json = ? WHERE id = ?",
                        (update["text"], update["start"], update["end"], json.dumps(update["flags"]), sid),
                    )

            return self._record(conn, project_id, "recheck", summary, set(), set(updates), mutate, allow_job=job_id)

    # ------------------------------------------------------------------ history

    def undo(self, project_id: str) -> str | None:
        with self.db.transaction() as conn:
            self._ensure_editable(conn, project_id)
            row = conn.execute(
                "SELECT * FROM edits WHERE project_id = ? AND undone = 0 ORDER BY id DESC LIMIT 1", (project_id,)
            ).fetchone()
            if row is None:
                return None
            _restore(conn, project_id, json.loads(row["before_json"]))
            conn.execute("UPDATE edits SET undone = 1 WHERE id = ?", (row["id"],))
            _bump(conn, project_id)
            return row["summary"]

    def redo(self, project_id: str) -> str | None:
        with self.db.transaction() as conn:
            self._ensure_editable(conn, project_id)
            row = conn.execute(
                "SELECT * FROM edits WHERE project_id = ? AND undone = 1 ORDER BY id ASC LIMIT 1", (project_id,)
            ).fetchone()
            if row is None:
                return None
            _restore(conn, project_id, json.loads(row["after_json"]))
            conn.execute("UPDATE edits SET undone = 0 WHERE id = ?", (row["id"],))
            _bump(conn, project_id)
            return row["summary"]

    def history(self, project_id: str, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT id, kind, summary, undone, created_at FROM edits WHERE project_id = ? ORDER BY id DESC LIMIT ?",
            (project_id, int(limit)),
        )
        return [{"id": r["id"], "kind": r["kind"], "summary": r["summary"], "undone": bool(r["undone"]),
                 "created_at": r["created_at"]} for r in rows]

    # ------------------------------------------------------------------ plumbing

    def _record(self, conn, project_id: str, kind: str, summary: str, speakers: set[int], segments: set[int],
                mutate: Callable, created_speakers: Callable[[], set[int]] = set,
                created_segments: Callable[[], set[int]] = set, allow_job: str | None = None) -> str:
        self._ensure_editable(conn, project_id, allow_job)
        before = _capture(conn, speakers, segments)
        mutate(conn)
        new_speakers, new_segments = created_speakers(), created_segments()
        after = _capture(conn, speakers | new_speakers, segments | new_segments)
        for s in new_speakers:
            before["speakers"][str(s)] = None
        for s in new_segments:
            before["segments"][str(s)] = None
        conn.execute("DELETE FROM edits WHERE project_id = ? AND undone = 1", (project_id,))
        conn.execute(
            """INSERT INTO edits (project_id, kind, summary, before_json, after_json, undone, created_at)
               VALUES (?, ?, ?, ?, ?, 0, ?)""",
            (project_id, kind, summary, json.dumps(before), json.dumps(after), utc_now()),
        )
        _bump(conn, project_id)
        return summary

    @staticmethod
    def _ensure_editable(conn, project_id: str, allow_job: str | None = None) -> None:
        if conn.execute("SELECT 1 FROM projects WHERE id = ?", (project_id,)).fetchone() is None:
            raise NotFound("Project not found.", data={"project_id": project_id})
        active = conn.execute(
            "SELECT 1 FROM jobs WHERE project_id = ? AND status IN ('queued','running') AND id IS NOT ? LIMIT 1",
            (project_id, allow_job),
        ).fetchone()
        if active:
            raise InvalidJobState(
                "This project is being transcribed; its new transcript will replace the current one. "
                "Wait for it to finish (or cancel it) before making corrections."
            )

    @staticmethod
    def _speaker(conn, project_id: str, speaker_id: int):
        row = conn.execute("SELECT * FROM speakers WHERE id = ? AND project_id = ?", (int(speaker_id), project_id)).fetchone()
        if row is None:
            raise NotFound("Speaker not found.", data={"speaker_id": speaker_id})
        return row

    @staticmethod
    def _segment(conn, project_id: str, segment_id: int):
        row = conn.execute("SELECT * FROM segments WHERE id = ? AND project_id = ?", (int(segment_id), project_id)).fetchone()
        if row is None:
            raise NotFound("Segment not found.", data={"segment_id": segment_id})
        return row

    @staticmethod
    def _words(conn, segment_id: int) -> list:
        return conn.execute("SELECT * FROM words WHERE segment_id = ? ORDER BY ordinal", (segment_id,)).fetchall()

    @staticmethod
    def _display(row) -> str:
        return row["name"] or row["label"]

    @staticmethod
    def _segments_mentioning(conn, project_id: str, speakers: set[int]) -> set[int]:
        out = set()
        for r in conn.execute("SELECT id, speaker_id, flags_json FROM segments WHERE project_id = ?", (project_id,)):
            if r["speaker_id"] in speakers:
                out.add(r["id"])
                continue
            flags = json.loads(r["flags_json"])
            if any(set(f.get("other_speakers", [])) & speakers for f in flags):
                out.add(r["id"])
        marks = ",".join("?" * len(speakers))
        for r in conn.execute(
            f"""SELECT DISTINCT segment_id FROM words WHERE speaker_id IN ({marks})
                AND segment_id IN (SELECT id FROM segments WHERE project_id = ?)""",
            (*speakers, project_id),
        ):
            out.add(r["segment_id"])
        return out

    @staticmethod
    def _insert_speaker(conn, project_id: str, name: str | None) -> int:
        numbers = [int(m.group(1)) for (label,) in conn.execute("SELECT label FROM speakers WHERE project_id = ?", (project_id,))
                   if (m := re.fullmatch(r"Speaker (\d+)", label))]
        count = conn.execute("SELECT COUNT(*) FROM speakers WHERE project_id = ?", (project_id,)).fetchone()[0]
        return _insert(conn, "speakers", {
            "project_id": project_id,
            "label": f"Speaker {max(numbers, default=0) + 1}",
            "name": (name or "").strip()[:MAX_NAME_LENGTH] or None,
            "origin": "user",
            "cluster_index": None,
            "color_index": count,
            "merged_into": None,
            "created_at": utc_now(),
        })


# --------------------------------------------------------------------------- word re-derivation


def rederive_words(old_words: list, new_text: str, seg_start: float, seg_end: float) -> list[dict[str, Any]]:
    """Words for an edited text, reusing the old words' timings where the
    words are unchanged (case and punctuation ignored) and interpolating
    times for inserted or rewritten words."""
    tokens = re.findall(r"\s*\S+", new_text)
    old = [dict(w) for w in old_words]
    a = [normalize_token(w["text"]) for w in old]
    b = [normalize_token(t) for t in tokens]
    out: list[dict[str, Any]] = []

    def blank(token: str, start: float | None, end: float | None) -> dict[str, Any]:
        return {"start": start, "end": end, "text": token, "probability": None, "timing_source": "interpolated",
                "whisper_start": None, "whisper_end": None, "alignment_score": None, "original_text": None}

    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                w = old[i1 + k]
                out.append({**_word_values(w), "text": tokens[j1 + k]})
        elif op == "delete":
            continue
        else:
            span_words = old[i1:i2]
            if op == "replace" and i2 - i1 == j2 - j1:
                for k in range(i2 - i1):  # same number of words: a correction of each word
                    w = old[i1 + k]
                    out.append({**_word_values(w), "text": tokens[j1 + k], "probability": None})
                continue
            # different number of words: spread them over the old words' time span
            # (or the gap between neighbours for a pure insertion)
            if span_words:
                start = min(w["start"] for w in span_words if w["start"] is not None) if any(w["start"] is not None for w in span_words) else None
                end = max(w["end"] for w in span_words if w["end"] is not None) if any(w["end"] is not None for w in span_words) else None
            else:
                prev_end = next((o["end"] for o in reversed(out) if o["end"] is not None), seg_start)
                following = next((w["start"] for w in old[i1:] if w["start"] is not None), seg_end)
                start, end = prev_end, max(prev_end, following)
            new_tokens = tokens[j1:j2]
            if start is None or end is None:
                out.extend(blank(t, start, end) for t in new_tokens)
                continue
            lengths = [max(1, len(t.strip())) for t in new_tokens]
            total = sum(lengths)
            t = start
            for token, length in zip(new_tokens, lengths):
                duration = (end - start) * length / total
                out.append(blank(token, round(t, 3), round(t + duration, 3)))
                t += duration
    return out


def _word_values(w: dict[str, Any]) -> dict[str, Any]:
    return {k: w[k] for k in _WORD_COLUMNS if k not in ("id", "segment_id", "ordinal", "speaker_id")}


def _join(texts: Iterable[str]) -> str:
    return "".join(texts)


def _join_segments(texts: Iterable[str]) -> str:
    out = ""
    for text in texts:
        if out and not out.endswith((" ", "\n")) and not text.startswith((" ", "\n")):
            out += " "
        out += text
    return out


# --------------------------------------------------------------------------- snapshots


def _capture(conn, speakers: set[int], segments: set[int]) -> dict[str, Any]:
    state: dict[str, Any] = {"speakers": {}, "segments": {}}
    for sid in speakers:
        row = conn.execute("SELECT * FROM speakers WHERE id = ?", (sid,)).fetchone()
        state["speakers"][str(sid)] = {k: row[k] for k in _SPEAKER_COLUMNS} if row else None
    for sid in segments:
        row = conn.execute("SELECT * FROM segments WHERE id = ?", (sid,)).fetchone()
        if row is None:
            state["segments"][str(sid)] = None
            continue
        words = conn.execute("SELECT * FROM words WHERE segment_id = ? ORDER BY ordinal", (sid,)).fetchall()
        state["segments"][str(sid)] = {
            **{k: row[k] for k in _SEGMENT_COLUMNS},
            "words": [{k: w[k] for k in _WORD_COLUMNS} for w in words],
        }
    return state


def _restore(conn, project_id: str, state: dict[str, Any]) -> None:
    for sid, row in state["speakers"].items():
        if row is not None:
            _upsert(conn, "speakers", row)
    for sid, row in state["segments"].items():
        if row is None:
            conn.execute("DELETE FROM segments WHERE id = ? AND project_id = ?", (int(sid), project_id))
            continue
        words = row["words"]
        _upsert(conn, "segments", {k: v for k, v in row.items() if k != "words"})
        conn.execute("DELETE FROM words WHERE segment_id = ?", (int(sid),))
        # words may currently belong to another segment (split/merge): free their ids
        for w in words:
            conn.execute("DELETE FROM words WHERE id = ?", (w["id"],))
        for w in words:
            _insert(conn, "words", w)
    for sid, row in state["speakers"].items():
        if row is None:
            conn.execute("DELETE FROM speakers WHERE id = ? AND project_id = ?", (int(sid), project_id))


def _insert(conn, table: str, values: dict[str, Any]) -> int:
    columns = list(values)
    cur = conn.execute(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
        [values[c] for c in columns],
    )
    return cur.lastrowid


def _upsert(conn, table: str, values: dict[str, Any]) -> None:
    columns = list(values)
    updates = ", ".join(f"{c} = excluded.{c}" for c in columns if c != "id")
    conn.execute(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))}) "
        f"ON CONFLICT(id) DO UPDATE SET {updates}",
        [values[c] for c in columns],
    )


def _bump(conn, project_id: str) -> None:
    conn.execute(
        "UPDATE projects SET transcript_revision = transcript_revision + 1, updated_at = ? WHERE id = ?",
        (utc_now(), project_id),
    )

