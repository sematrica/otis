"""Passes 6-7 as one procedure: plan, decode again, compare, apply.

Used by Maximum Accuracy jobs for every doubtful passage, and by on-demand
re-check jobs for the passages a user picks. Models are loaded one at a time:
the recogniser for the new decodes, then the acoustic model for comparing
them and re-timing changed words.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..alignment.aligner import WordAligner
from ..alignment.models import load_model as load_alignment_model
from ..alignment.score import PassageScorer
from ..asr.base import DecodeOptions, Word
from ..asr.registry import create_backend
from ..audio import SAMPLE_RATE
from ..audio.wav import WavReader
from ..transcript.build import apply_alignment
from ..transcript.model import TSegment
from ..util.hardware import performance_core_count
from .compare import Candidate, compare
from .quality import SpeechMap, segment_coverage
from .retranscribe import (
    Decision,
    apply_decisions,
    decode_configs,
    extract,
    load_decodes,
    plan_windows,
    prompt_before,
    save_decode,
    scoring_window,
    select_targets,
)

#: Silence before audio that starts at 0 s (see the worker's first chunk).
LEAD_IN_SECONDS = 0.5


@dataclass
class RecheckResult:
    segments: list[TSegment]
    #: segments whose words were replaced or added (final objects)
    changed: list[TSegment]
    counts: dict[str, int]
    info: dict[str, Any]
    #: re-worded segments by their index in the input (before speaker re-assignment)
    replaced: dict[int, TSegment]


class Recheck:
    def __init__(self, segments: list[TSegment], language: str | None, resolved, duration: float, *,
                 indices: set[int] | None = None, protected: set[int] | None = None,
                 gaps: Sequence[dict[str, Any]] = ()):
        self.segments = segments
        self.language = language
        self.resolved = resolved
        self.duration = duration
        self.protected = protected or set()
        self.skipped: list[tuple[int, str]] = []
        self.targets = select_targets(segments, indices, self.skipped, gaps)
        self.windows = plan_windows(self.targets, duration)
        self.words = sorted((w for s in segments for w in s.words), key=lambda w: w.start)
        self.decodes: list[tuple[int, str, DecodeOptions, str]] = []
        model_key = str(resolved.model_dir)
        for index, window in enumerate(self.windows):
            lang = self._language(window.targets[0])
            hallucination = any(t.hallucination for t in window.targets)
            for name, options in decode_configs(resolved.decode, lang, prompt_before(self.words, window.start),
                                                hallucination):
                self.decodes.append((index, name, options, window.key(model_key, name, options)))
        self.cached: dict[str, list[Word]] = {}

    def _language(self, target) -> str | None:
        seg = self.segments[target.segment] if target.segment is not None else None
        return (seg.language if seg is not None else None) or self.language

    # pass 6 --------------------------------------------------------------------

    def decode(self, reader: WavReader, cache_path: Path, *, progress: Callable[[int, int], None] | None = None,
               check_cancel: Callable[[], None] | None = None) -> None:
        """Decode every window with every configuration; results are appended
        to ``cache_path`` as they come, so an interrupted job resumes."""
        self.cached = load_decodes(cache_path)
        pending = [d for d in self.decodes if d[3] not in self.cached]
        done = len(self.decodes) - len(pending)
        if not pending:
            return
        backend = create_backend(self.resolved.backend)
        backend.load(self.resolved.model_dir, self.resolved.load)
        try:
            for index, _name, options, key in pending:
                if check_cancel is not None:
                    check_cancel()
                window = self.windows[index]
                audio = reader.read(window.start, window.end)
                lead = LEAD_IN_SECONDS if window.start <= 0.0 else 0.0
                if lead:
                    audio = np.concatenate([np.zeros(int(lead * SAMPLE_RATE), dtype=np.float32), audio])
                result, _ = backend.transcribe(audio, options, None)
                offset = window.start - lead
                decoded = [
                    Word(max(window.start, w.start + offset), max(window.start, w.end + offset), w.text, w.probability)
                    for seg in result.segments for w in (seg.words or [])
                ]
                save_decode(cache_path, key, decoded)
                self.cached[key] = decoded
                done += 1
                if progress is not None:
                    progress(done, len(self.decodes))
        finally:
            backend.close()

    # pass 7 and applying ----------------------------------------------------------

    def finish(self, reader: WavReader, *, assign: Callable[[TSegment], list[TSegment]] | None = None,
               reanalyse: Callable[[list[TSegment]], None] | None = None,
               speech: SpeechMap | None = None) -> RecheckResult:
        """Compare the candidates by acoustic evidence and apply the decisions.

        ``assign`` re-assigns speakers to a changed segment (it may split it);
        without it, changed segments keep their speaker. ``reanalyse`` re-runs
        the quality checks on the changed segments. ``speech`` (the speech
        map) tells where speech was detected independently of the acoustic
        model."""
        choice = self.resolved.scoring_model_for(self.language)
        model = load_alignment_model(choice[0], choice[1], performance_core_count()) if choice else None
        decisions: list[Decision] = []
        for i, reason in self.skipped:
            self.segments[i].flags.append({"type": "rechecked", "result": "not_rechecked", "reason": reason})
        for index, window in enumerate(self.windows):
            context = [w for w in self.words if w.end > window.start and w.start < window.end]
            claimed: dict[str, set[int]] = {}
            for target in window.targets:
                first = [] if target.gap else list(self.segments[target.segment].words)
                candidates = [Candidate("first_pass", [w.text for w in first])]
                timed = [[Word(w.start, w.end, w.text, w.probability) for w in first]]
                for d_index, name, _options, key in self.decodes:
                    if d_index == index:
                        found = extract(context, target, self.cached.get(key, []), claimed.setdefault(key, set()))
                        candidates.append(Candidate(name, [w.text for w in found]))
                        timed.append(found)
                comparison = None
                language = self._language(target)
                if model is not None and model.supports(language):
                    scorer = PassageScorer(model, language)
                    sw = scoring_window(self.words, target, self.duration)
                    comparison = compare(scorer, scorer.emissions(reader.read(sw.start, sw.end)), candidates,
                                         left=sw.left, right=sw.right)
                decisions.append(Decision(target, candidates, timed, comparison))
        segments, changed_idx, counts, replaced = apply_decisions(
            self.segments, decisions, protected=self.protected, language=self.language,
            speech_coverage=(lambda seg: segment_coverage(seg, speech)) if speech is not None else None)
        changed = [segments[i] for i in sorted(changed_idx)]
        if changed and model is not None:
            apply_alignment(changed, WordAligner(model).align(reader, changed, self.language))
        del model
        if changed and assign is not None:
            rebuilt: list[TSegment] = []
            fresh: list[TSegment] = []
            ids = {id(s) for s in changed}
            for seg in segments:
                if id(seg) in ids:
                    parts = assign(seg)
                    rebuilt.extend(parts)
                    fresh.extend(parts)
                else:
                    rebuilt.append(seg)
            segments, changed = rebuilt, fresh
        if changed and reanalyse is not None:
            reanalyse(changed)
        info = {
            "targets": len(self.targets),
            "skipped_overlapping_speech": len(self.skipped),
            "windows": len(self.windows),
            "decodes": len(self.decodes),
            "acoustic_model": None if choice is None else {"id": choice[0].id, "display_name": choice[0].display_name},
        }
        return RecheckResult(segments, changed, counts, info, replaced)


def summary_text(counts: dict[str, int], has_model: bool) -> str:
    parts = [f"{counts.get('replaced', 0)} re-worded", f"{counts.get('recovered', 0)} recovered",
             f"{counts.get('unresolved', 0)} with alternatives",
             f"{counts.get('no_acoustic_support', 0)} without acoustic support"]
    text = ", ".join(p for p in parts if not p.startswith("0 ")) or "First-pass wording kept everywhere."
    if not has_model:
        text += " No acoustic model for this language: alternatives listed, nothing changed."
    return text


def segments_spans(segments: Sequence[TSegment]) -> list[tuple[float, float]]:
    return [(s.start, s.end) for s in segments]
