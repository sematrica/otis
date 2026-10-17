"""Pass 6: targeted re-transcription of doubtful passages.

Passages flagged by quality analysis (unclear speech, possible hallucination,
failed alignment, untranscribed speech) are decoded again from the original
recording, each in a window with :data:`PAD_SECONDS` of context on both
sides, with settings that differ from the first pass in ways that matter for
the usual failure modes:

* ``context`` - deterministic beam search, the preceding transcript as a
  prompt (names, terms), no conditioning loops, no VAD (nothing is skipped);
* ``plain`` - the same without the prompt (a prompt can also bias);
* ``vad`` - only for suspected hallucinations: VAD on, Whisper's
  hallucination-silence check on.

From each decode, the words that correspond to the target passage are cut out
by aligning the decode with the first pass's surrounding words. The resulting
candidates go to pass 7 (:mod:`.compare`), which decides by acoustic
evidence, and :func:`apply_decisions` updates the transcript, recording every
decision in the segment's flags (first-pass text, candidates, scores).
"""

from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from ..asr.base import DecodeOptions, Word
from ..asr.merge import normalize_token
from ..transcript.model import TSegment, TWord
from .compare import DECISION_MARGIN, Candidate, Comparison
from .review import LEVELS

#: Context decoded on each side of a doubtful passage.
PAD_SECONDS = 2.0
#: Passages whose padded windows overlap are decoded together up to this length.
MAX_WINDOW_SECONDS = 28.0
#: Flags that make a segment a re-transcription target (at this severity or more).
TARGET_FLAGS = {"low_confidence": "medium", "possible_hallucination": "medium", "alignment_failed": "low",
                "missing_speech": "medium"}
#: Anchor words (shared by all candidates) must be this close to the passage.
ANCHOR_MAX_GAP = 1.5
ANCHOR_WORDS = 2
#: Silence margin of the scoring window where there is no anchor word.
EDGE_SECONDS = 0.5
#: Decoded and first-pass words "match" only if this close in time.
MATCH_TOLERANCE = 1.5
PROMPT_WORDS = 48
#: A passage overlapping another segment by this much is overlapping speech:
#: one audio stream decoded again cannot separate the speakers, and acoustic
#: scores there mix both voices, so it is not re-worded automatically.
OVERLAP_SKIP_SECONDS = 0.25
#: Speaker and pass-5 flags are recomputed for changed segments.
RECOMPUTED_FLAGS = {"low_confidence", "possible_hallucination", "repeated_words", "overlapping_speech",
                    "speaker_uncertain", "speaker_change_uncertain", "speaker_unassigned", "split_at_speaker_change",
                    "alignment_failed"}


@dataclass
class Target:
    start: float
    end: float
    #: index of the segment re-checked, or None for untranscribed speech
    segment: int | None
    reasons: list[str] = field(default_factory=list)
    hallucination: bool = False

    @property
    def gap(self) -> bool:
        return self.segment is None


@dataclass
class Window:
    start: float
    end: float
    targets: list[Target]

    def key(self, model: str, config: str, options: DecodeOptions) -> str:
        digest = hashlib.sha256(f"{model}|{config}|{self.start:.3f}|{self.end:.3f}|{options!r}".encode())
        return digest.hexdigest()[:20]


def _span(seg: TSegment) -> tuple[float, float]:
    if seg.words:
        return seg.words[0].start, seg.words[-1].end
    return seg.start, seg.end


def _overlaps_another(segments: Sequence[TSegment], i: int) -> bool:
    start, end = _span(segments[i])
    for j, other in enumerate(segments):
        if j == i or not other.words:
            continue
        a, b = _span(other)
        if min(end, b) - max(start, a) >= OVERLAP_SKIP_SECONDS:
            return True
    return False


def select_targets(segments: Sequence[TSegment], indices: set[int] | None = None,
                   skipped: list[tuple[int, str]] | None = None,
                   gaps: Sequence[dict[str, Any]] = ()) -> list[Target]:
    """Doubtful passages (or exactly ``indices``, when the user asks).
    Passages in overlapping speech are left out and listed in ``skipped``.
    ``gaps`` are stretches of untranscribed speech not attached to a segment
    (the transcript is empty)."""
    targets: list[Target] = []
    if indices is None:
        for gap in gaps:
            if LEVELS.get(gap.get("severity"), 0) >= LEVELS[TARGET_FLAGS["missing_speech"]]:
                targets.append(Target(float(gap["start"]), float(gap["end"]), None, ["missing_speech"]))
    for i, seg in enumerate(segments):
        reasons: list[str] = []
        hallucination = False
        for flag in seg.flags:
            kind = flag.get("type")
            if kind == "missing_speech":
                if indices is None and LEVELS.get(flag.get("severity"), 0) >= LEVELS[TARGET_FLAGS[kind]]:
                    targets.append(Target(float(flag["start"]), float(flag["end"]), None, ["missing_speech"]))
                continue
            if kind not in TARGET_FLAGS:
                continue
            severity = flag.get("severity") or "medium"
            if LEVELS.get(severity, 0) >= LEVELS[TARGET_FLAGS[kind]]:
                reasons.append(kind)
                hallucination |= kind == "possible_hallucination"
        wanted = (i in indices) if indices is not None else bool(reasons)
        if not wanted or not seg.words:
            continue
        if _overlaps_another(segments, i):
            if skipped is not None:
                skipped.append((i, "overlapping speech"))
            continue
        start, end = _span(seg)
        targets.append(Target(start, end, i, reasons or ["requested"], hallucination))
    targets.sort(key=lambda t: t.start)
    return targets


def plan_windows(targets: Sequence[Target], duration: float) -> list[Window]:
    windows: list[Window] = []
    for target in sorted(targets, key=lambda t: t.start):
        start = max(0.0, target.start - PAD_SECONDS)
        end = min(duration, target.end + PAD_SECONDS)
        if windows and start <= windows[-1].end and end - windows[-1].start <= MAX_WINDOW_SECONDS:
            windows[-1].end = max(windows[-1].end, end)
            windows[-1].targets.append(target)
        else:
            windows.append(Window(start, end, [target]))
    return windows


def decode_configs(base: DecodeOptions, language: str | None, prompt: str | None,
                   hallucination: bool) -> list[tuple[str, DecodeOptions]]:
    common = dict(language=language, word_timestamps=True, condition_on_previous_text=False,
                  beam_size=max(5, base.beam_size), best_of=max(5, base.best_of))
    configs = [
        ("context", replace(base, **common, temperatures=(0.0,), initial_prompt=prompt, vad=False)),
        ("plain", replace(base, **common, temperatures=(0.0,), initial_prompt=None, vad=False)),
    ]
    if hallucination:
        configs.append(("vad", replace(base, **common, initial_prompt=None, vad=True,
                                       hallucination_silence_threshold=1.0)))
    return configs


def prompt_before(words: Sequence[TWord], start: float) -> str | None:
    before = [w.text.strip() for w in words if w.end <= start]
    return " ".join(before[-PROMPT_WORDS:]) or None


# ----------------------------------------------------------------------------- candidates


def extract(context: Sequence[TWord], target: Target, decoded: Sequence[Word],
            claimed: set[int] | None = None) -> list[Word]:
    """The decoded words that stand for the target passage.

    ``context`` are the first pass's words in the window. The decode is
    aligned with them; the last matching word before the passage and the first
    matching word after it delimit the candidate. Without a match on a side,
    word midpoints decide on that side. ``claimed`` holds indices of decoded
    words already given to another passage of the window: a word stands for
    one passage only (they are added to it)."""
    left = [w for w in context if (w.start + w.end) / 2 < target.start]
    inner = [w for w in context if target.start <= (w.start + w.end) / 2 <= target.end]
    right = [w for w in context if (w.start + w.end) / 2 > target.end]
    reference = [*left, *inner, *right]
    a = [normalize_token(w.text) for w in reference]
    b = [normalize_token(w.text) for w in decoded]
    cut_left: int | None = None
    cut_right: int | None = None
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            i, j = block.a + k, block.b + k
            if not a[i] or abs(reference[i].start - decoded[j].start) > MATCH_TOLERANCE:
                continue
            if i < len(left):
                cut_left = j + 1
            elif i >= len(left) + len(inner) and cut_right is None:
                cut_right = j
    if cut_left is None:
        cut_left = next((j for j, w in enumerate(decoded) if (w.start + w.end) / 2 >= target.start - 0.15), len(decoded))
    if cut_right is None:
        cut_right = next((j for j, w in enumerate(decoded) if (w.start + w.end) / 2 > target.end + 0.15), len(decoded))
    if cut_right < cut_left:
        return []
    picked = [j for j in range(cut_left, cut_right) if claimed is None or j not in claimed]
    if claimed is not None:
        claimed.update(picked)
    return [decoded[j] for j in picked]


@dataclass
class ScoringWindow:
    start: float
    end: float
    left: list[str]
    right: list[str]


def scoring_window(words: Sequence[TWord], target: Target, duration: float) -> ScoringWindow:
    """Audio and anchor words for comparing candidates of one passage: up to
    :data:`ANCHOR_WORDS` transcript words on each side, if close enough."""
    before = [w for w in words if w.end <= target.start + 0.05 and (w.start + w.end) / 2 < target.start]
    after = [w for w in words if w.start >= target.end - 0.05 and (w.start + w.end) / 2 > target.end]
    left = [w for w in before[-ANCHOR_WORDS:] if target.start - w.end <= ANCHOR_MAX_GAP]
    right = [w for w in after[:ANCHOR_WORDS] if w.start - target.end <= ANCHOR_MAX_GAP]
    if left:
        start = left[0].start - 0.1
    else:
        start = max(target.start - EDGE_SECONDS, before[-1].end if before else 0.0)
    if right:
        end = right[-1].end + 0.1
    else:
        end = min(target.end + EDGE_SECONDS, after[0].start if after else duration)
    return ScoringWindow(max(0.0, start), min(duration, end), [w.text for w in left], [w.text for w in right])


# ----------------------------------------------------------------------------- applying decisions


@dataclass
class Decision:
    target: Target
    candidates: list[Candidate]
    #: decoded words (with Whisper timings) of each candidate, parallel to candidates
    words: list[list[Word]]
    comparison: Comparison | None  # None: no acoustic model for this language
    sources: list[str] = field(default_factory=list)

    def candidates_payload(self) -> list[dict[str, Any]]:
        scores = {s.source: s.log_likelihood for s in self.comparison.scores} if self.comparison else {}
        seen: set[tuple[str, ...]] = set()
        out = []
        for cand in self.candidates:
            if cand.key in seen:
                continue
            seen.add(cand.key)
            score = scores.get(cand.source)
            out.append({"source": cand.source, "text": join_words(cand.words),
                        "score": None if score is None else round(score, 3)})
        return out


def join_words(words: Sequence[str]) -> str:
    """Text of decoded words, which carry their own leading spaces (none
    between words in languages written without spaces)."""
    return " ".join("".join(words).split())


def _distinct_wordings(decision: Decision) -> int:
    return len({c.key for c in decision.candidates if c.key or decision.target.gap})


def apply_decisions(segments: list[TSegment], decisions: Sequence[Decision], *, protected: set[int] | None = None,
                    language: str | None = None) -> tuple[list[TSegment], set[int], dict[str, int], dict[int, TSegment]]:
    """Apply pass-7 decisions. Returns the new segment list, the indices (in
    it) of segments whose words changed or were added, counts per outcome, and
    the replacement of each re-worded segment by its original index.

    Segments in ``protected`` (text corrected by the user) are never re-worded:
    a better-supported wording is offered as an alternative instead."""
    protected = protected or set()
    counts = {"replaced": 0, "recovered": 0, "unresolved": 0, "no_acoustic_support": 0, "confirmed": 0,
              "same_wording": 0, "not_comparable": 0}
    replaced: dict[int, TSegment] = {}
    inserted: list[TSegment] = []
    for decision in decisions:
        target, comparison = decision.target, decision.comparison
        first_text = join_words(decision.candidates[0].words)
        payload = decision.candidates_payload()
        seg = segments[target.segment] if target.segment is not None else None
        earlier = None
        if seg is not None:
            # A passage checked again: the new comparison replaces the old one.
            earlier = next((f for f in seg.flags if f.get("type") == "retranscribed"), None)
            seg.flags[:] = [f for f in seg.flags if f.get("type") not in ("rechecked", "alternatives")]
        if comparison is None or comparison.decision == "unscorable":
            outcome = "not_comparable"
            reason = "no acoustic model for this language" if comparison is None else "words the acoustic model cannot spell"
            if seg is not None:
                seg.flags.append({"type": "rechecked", "result": outcome, "reason": reason})
                if _distinct_wordings(decision) > 1:
                    seg.flags.append({"type": "alternatives", "severity": "medium", "candidates": payload,
                                      "reason": reason})
            counts[outcome] += 1
            continue
        if comparison.decision == "replace" and target.segment in protected:
            seg.flags.append({"type": "alternatives", "severity": "medium", "candidates": payload,
                              "margin": comparison.margin, "reason": "your correction is kept"})
            seg.flags.append({"type": "rechecked", "result": "alternative_to_correction"})
            counts["unresolved"] += 1
            continue
        if comparison.decision == "replace":
            chosen = decision.words[comparison.chosen]
            source = decision.candidates[comparison.chosen].source
            speaker = seg.speaker if seg is not None else None
            new_words = [TWord(w.start, w.end, w.text, w.probability, "whisper", w.start, w.end, speaker=speaker)
                         for w in chosen]
            if seg is not None:
                updated = TSegment(
                    start=min(seg.start, new_words[0].start) if new_words else seg.start,
                    end=max(seg.end, new_words[-1].end) if new_words else seg.end,
                    text="".join(w.text for w in new_words).strip(),
                    words=new_words, language=seg.language, chunk_index=seg.chunk_index,
                    flags=[f for f in seg.flags if f.get("type") not in RECOMPUTED_FLAGS | {"retranscribed"}],
                    speaker=seg.speaker, speaker_confidence=seg.speaker_confidence,
                )
                # "first_pass" stays the recogniser's original wording
                original = earlier["first_pass"] if earlier is not None else first_text
                updated.flags.append({"type": "retranscribed", "severity": "low", "first_pass": original,
                                      "chosen": source, "margin": comparison.margin, "candidates": payload})
                replaced[target.segment] = updated
                counts["replaced"] += 1
            else:
                new = TSegment(start=new_words[0].start, end=new_words[-1].end,
                               text="".join(w.text for w in new_words).strip(), words=new_words,
                               language=segments[0].language if segments else language)
                new.flags.append({"type": "recovered_speech", "severity": "medium", "gap": [target.start, target.end],
                                  "chosen": source, "margin": comparison.margin, "candidates": payload})
                inserted.append(new)
                counts["recovered"] += 1
            continue
        if seg is None:  # untranscribed speech that stays untranscribed
            counts["unresolved" if comparison.decision == "unresolved" else "same_wording"] += 1
            continue
        if comparison.decision == "no_acoustic_support":
            flag = next((f for f in seg.flags if f.get("type") == "possible_hallucination"), None)
            if flag is None:
                flag = {"type": "possible_hallucination", "reasons": []}
                seg.flags.append(flag)
            flag["severity"] = "high"
            flag["reasons"] = [*(r for r in flag.get("reasons", []) if r != "no_acoustic_support"), "no_acoustic_support"]
            flag["acoustic_margin"] = comparison.margin
            seg.flags.append({"type": "rechecked", "result": "no_acoustic_support", "candidates": payload})
            counts["no_acoustic_support"] += 1
        elif comparison.decision == "unresolved":
            seg.flags.append({"type": "alternatives", "severity": "medium", "candidates": payload,
                              "margin": comparison.margin})
            seg.flags.append({"type": "rechecked", "result": "unresolved"})
            counts["unresolved"] += 1
        elif comparison.margin is not None and comparison.margin >= DECISION_MARGIN:
            # Other wordings were tried and the audio clearly supports the first pass.
            for flag in seg.flags:
                if flag.get("type") == "low_confidence":
                    flag["severity"] = "low"
                    flag["confirmed_acoustically"] = True
            seg.flags.append({"type": "rechecked", "result": "confirmed", "margin": comparison.margin,
                              "candidates": payload})
            counts["confirmed"] += 1
        else:
            # Every decode produced the same words: agreement is not evidence.
            seg.flags.append({"type": "rechecked", "result": "same_wording"})
            counts["same_wording"] += 1

    out: list[TSegment] = []
    changed: set[int] = set()
    for i, seg in enumerate(segments):
        out.append(replaced.get(i, seg))
    out.extend(inserted)
    out.sort(key=lambda s: s.start)
    for i, seg in enumerate(out):
        if any(seg is s for s in replaced.values()) or any(seg is s for s in inserted):
            changed.add(i)
    return out, changed, counts, replaced


def save_decode(path, key: str, words: Sequence[Word]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps({"key": key, "words": [[w.start, w.end, w.text, w.probability] for w in words]}) + "\n")


def load_decodes(path) -> dict[str, list[Word]]:
    out: dict[str, list[Word]] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        try:
            entry = json.loads(line)
            out[entry["key"]] = [Word(*w) for w in entry["words"]]
        except (ValueError, KeyError, TypeError):
            continue
    return out
