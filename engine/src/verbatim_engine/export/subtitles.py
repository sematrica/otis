"""SRT and WebVTT subtitles.

Cues are cut from the words' own times (forced alignment where available), so
each subtitle appears while its words are spoken. A cue never spans two
segments, so it never mixes speakers. Limits follow common subtitle practice:
at most two lines of 42 characters and 7 seconds per cue; a cue also ends at
the end of a sentence once it has run for 1.5 seconds. A cue shorter than
0.7 s is held on screen up to 0.7 s, but never into the next cue. Segments
without word times are split evenly by characters over the segment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..util.timecode import format_precise
from .common import speaker_names, spoken_segments

MAX_LINE = 42
MAX_LINES = 2
MAX_SECONDS = 7.0
SENTENCE_SECONDS = 1.5
MIN_SECONDS = 0.7
SENTENCE_END = (".", "?", "!", "…", "。", "？", "！")


@dataclass
class Cue:
    start: float
    end: float
    text: str
    speaker: str | None
    #: first cue of a speaker turn
    turn_start: bool


def _timed_words(seg: dict[str, Any]) -> list[tuple[float, float, str]]:
    words = [w for w in seg.get("words") or [] if w["text"].strip()]
    if words and all(w["start"] is not None and w["end"] is not None for w in words):
        return [(w["start"], max(w["end"], w["start"]), w["text"].strip()) for w in words]
    # no word times: spread the text's words over the segment by length
    tokens = seg["text"].split()
    total = sum(len(t) + 1 for t in tokens) or 1
    span = max(0.0, seg["end"] - seg["start"])
    out, t = [], seg["start"]
    for i, token in enumerate(tokens):
        end = seg["end"] if i == len(tokens) - 1 else t + span * (len(token) + 1) / total
        out.append((t, end, token))
        t = end
    return out


def wrap(text: str, width: int = MAX_LINE) -> list[str]:
    """Balanced lines of at most ``width`` characters where possible."""
    if len(text) <= width:
        return [text]
    words = text.split()
    best: tuple[int, list[str]] | None = None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        cost = max(len(a), len(b))
        if best is None or cost < best[0]:
            best = (cost, [a, b])
    return best[1] if best else [text]


def _fits(words: list[str], prefix: str = "") -> bool:
    text = prefix + " ".join(words)
    if len(text) <= MAX_LINE:
        return True
    lines = wrap(text)
    return len(lines) <= MAX_LINES and all(len(line) <= MAX_LINE for line in lines)


def cues(transcript: dict[str, Any], speakers: bool = True, label_turns: bool = False) -> list[Cue]:
    """Cues of the transcript. With ``label_turns`` the first cue of each
    speaker turn will carry "Name: " (SRT), which counts towards its length."""
    names = speaker_names(transcript) if speakers else {}
    out: list[Cue] = []
    previous_speaker = object()
    for seg in spoken_segments(transcript):
        speaker = names.get(seg.get("speaker_id")) if names else None
        turn = speaker != previous_speaker
        previous_speaker = speaker
        current: list[tuple[float, float, str]] = []
        for word in _timed_words(seg):
            if current:
                texts = [w[2] for w in current] + [word[2]]
                prefix = f"{speaker}: " if label_turns and speaker and turn else ""
                too_long = word[1] - current[0][0] > MAX_SECONDS or not _fits(texts, prefix)
                sentence = (current[-1][2].endswith(SENTENCE_END)
                            and current[-1][1] - current[0][0] >= SENTENCE_SECONDS)
                if too_long or sentence:
                    out.append(Cue(current[0][0], current[-1][1], " ".join(w[2] for w in current), speaker, turn))
                    turn = False
                    current = []
            current.append(word)
        if current:
            out.append(Cue(current[0][0], current[-1][1], " ".join(w[2] for w in current), speaker, turn))
    out.sort(key=lambda c: c.start)
    for cue, following in zip(out, out[1:] + [None]):
        if cue.end - cue.start < MIN_SECONDS:
            limit = following.start if following is not None else cue.start + MIN_SECONDS
            cue.end = max(cue.end, min(cue.start + MIN_SECONDS, limit))
    return out


def render_srt(project: dict[str, Any], transcript: dict[str, Any], options: dict[str, Any]) -> str:
    blocks = []
    for index, cue in enumerate(cues(transcript, options.get("speakers", True), label_turns=True), start=1):
        text = cue.text
        if cue.speaker and cue.turn_start:
            text = f"{cue.speaker}: {text}"
        lines = wrap(text) if len(text) > MAX_LINE else [text]
        blocks.append(f"{index}\n{format_precise(cue.start, ',')} --> {format_precise(cue.end, ',')}\n"
                      + "\n".join(lines))
    return "\n\n".join(blocks) + "\n" if blocks else ""


def _vtt_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_vtt(project: dict[str, Any], transcript: dict[str, Any], options: dict[str, Any]) -> str:
    out = ["WEBVTT", ""]
    title = project["title"].replace("-->", "→").replace("\n", " ")
    out += [f"NOTE {title}", ""]
    for cue in cues(transcript, options.get("speakers", True)):
        lines = wrap(cue.text) if len(cue.text) > MAX_LINE else [cue.text]
        body = "\n".join(_vtt_escape(line) for line in lines)
        if cue.speaker:
            voice = _vtt_escape(cue.speaker).replace("\n", " ")
            body = f"<v {voice}>{body}</v>"
        out += [f"{format_precise(cue.start)} --> {format_precise(cue.end)}", body, ""]
    return "\n".join(out)
