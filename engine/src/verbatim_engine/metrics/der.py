"""Diarization error rate (DER).

Frame-based (10 ms) implementation of the standard NIST definition:

    DER = (missed speech + false alarm + speaker confusion) / reference speech

where every quantity counts speaker-seconds (two people talking at once count
twice), hypothesis speakers are mapped one-to-one onto reference speakers to
maximise their overlap (Hungarian algorithm), and, optionally, a ``collar``
around every reference boundary is not scored. With ``collar=0`` and
overlap scored this is the strict setting used by pyannote's benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

Turn = tuple[float, float, str]


def parse_rttm(text: str) -> dict[str, list[Turn]]:
    turns: dict[str, list[Turn]] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        uri, start, duration, label = parts[1], float(parts[3]), float(parts[4]), parts[7]
        turns.setdefault(uri, []).append((start, start + duration, label))
    return turns


@dataclass
class DERResult:
    der: float
    missed: float
    false_alarm: float
    confusion: float
    total: float
    mapping: dict[str, str]
    reference_speakers: int
    hypothesis_speakers: int

    def as_dict(self) -> dict:
        return {
            "der": round(self.der, 4),
            "missed": round(self.missed, 3),
            "false_alarm": round(self.false_alarm, 3),
            "confusion": round(self.confusion, 3),
            "total": round(self.total, 3),
            "reference_speakers": self.reference_speakers,
            "hypothesis_speakers": self.hypothesis_speakers,
            "mapping": self.mapping,
        }


def _matrix(turns: Iterable[Turn], labels: list[str], frames: int, resolution: float) -> np.ndarray:
    m = np.zeros((frames, len(labels)), dtype=bool)
    index = {label: i for i, label in enumerate(labels)}
    for start, end, label in turns:
        a, b = int(round(start / resolution)), int(round(end / resolution))
        m[max(0, a) : max(0, min(frames, b)), index[label]] = True
    return m


def diarization_error_rate(
    reference: list[Turn],
    hypothesis: list[Turn],
    *,
    collar: float = 0.0,
    skip_overlap: bool = False,
    scored_region: tuple[float, float] | None = None,
    resolution: float = 0.01,
) -> DERResult:
    from scipy.optimize import linear_sum_assignment

    end = max([t[1] for t in reference] + [t[1] for t in hypothesis] + [scored_region[1] if scored_region else 0.0])
    frames = int(np.ceil(end / resolution)) + 1
    ref_labels = sorted({t[2] for t in reference})
    hyp_labels = sorted({t[2] for t in hypothesis})
    ref = _matrix(reference, ref_labels, frames, resolution)
    hyp = _matrix(hypothesis, hyp_labels, frames, resolution)

    scored = np.ones(frames, dtype=bool)
    if scored_region is not None:
        scored[:] = False
        scored[int(round(scored_region[0] / resolution)) : int(round(scored_region[1] / resolution))] = True
    if collar > 0:
        c = int(round(collar / resolution))
        for start, stop, _ in reference:
            for t in (start, stop):
                f = int(round(t / resolution))
                scored[max(0, f - c) : f + c] = False
    if skip_overlap:
        scored &= ref.sum(axis=1) < 2
    ref, hyp = ref[scored], hyp[scored]

    mapping: dict[str, str] = {}
    correct = np.zeros(ref.shape[0], dtype=np.int64)
    if ref_labels and hyp_labels:
        cooccurrence = ref.T.astype(np.int64) @ hyp.astype(np.int64)
        rows, cols = linear_sum_assignment(cooccurrence, maximize=True)
        for r, h in zip(rows, cols):
            if cooccurrence[r, h] > 0:
                mapping[hyp_labels[h]] = ref_labels[r]
                correct += ref[:, r] & hyp[:, h]
    n_ref = ref.sum(axis=1)
    n_hyp = hyp.sum(axis=1)
    missed = np.maximum(n_ref - n_hyp, 0).sum() * resolution
    false_alarm = np.maximum(n_hyp - n_ref, 0).sum() * resolution
    confusion = (np.minimum(n_ref, n_hyp) - correct).sum() * resolution
    total = n_ref.sum() * resolution
    der = (missed + false_alarm + confusion) / total if total > 0 else 0.0
    return DERResult(float(der), float(missed), float(false_alarm), float(confusion), float(total), mapping, len(ref_labels), len(hyp_labels))
