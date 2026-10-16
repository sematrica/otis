"""Diarization pipeline: segmentation -> embeddings -> clustering -> timeline.

The output is a frame-level timeline on the segmentation model's grid
(16.875 ms frames) with, for every frame:

* how many speakers talk (``count``; 2 or more = overlapping speech),
* for every global speaker, the fraction of overlapping analysis windows that
  attribute the frame to them (``activation``, 0-1: temporal agreement),
* how clearly the embeddings behind that attribution favoured this speaker
  over the next most similar one (``margin``, cosine similarity gap).

Global speakers are numbered by first appearance.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..audio import SAMPLE_RATE
from ..audio.wav import WavReader
from ..errors import Cancelled
from . import clustering
from .clustering import UNASSIGNED, SpeakerBounds
from .embedding import (
    EMBEDDING_DIM,
    FBANK_SHIFT_SAMPLES,
    EmbeddingModel,
    filterbank,
    masked_embedding,
    window_features,
)
from .segmentation import (
    FRAME_DURATION,
    FRAME_STEP,
    FRAMES_PER_WINDOW,
    NUM_LOCAL_SPEAKERS,
    WINDOW_SAMPLES,
    SegmentationModel,
    WindowPlan,
    plan_windows,
    segment,
    total_frames,
    window_frame_offset,
)

log = logging.getLogger(__name__)

DEFAULT_STEP_SECONDS = 1.0
#: Phase weights for the progress bar, measured on CPU: segmentation is about
#: a fifth of the work, embeddings most of the rest.
_PHASE_WEIGHTS = {"segmentation": 0.2, "embeddings": 0.75, "clustering": 0.05}

ProgressFn = Callable[[float], None]


@dataclass
class DiarizationResult:
    duration: float
    activation: np.ndarray  # (T, K) float32
    count: np.ndarray  # (T,) uint8
    margin: np.ndarray  # (T, K) float32, NaN where no embedding evidence
    binary: np.ndarray  # (T, K) uint8: final per-speaker activity (overlap allowed)
    centroids: np.ndarray  # (K, 256)
    info: dict[str, Any] = field(default_factory=dict)

    frame_step: float = FRAME_STEP
    frame_offset: float = FRAME_DURATION / 2

    @property
    def num_speakers(self) -> int:
        return int(self.binary.shape[1])

    @property
    def frames(self) -> int:
        return int(self.binary.shape[0])

    def frame_bounds(self, index: int) -> tuple[float, float]:
        centre = index * self.frame_step + self.frame_offset
        return centre - self.frame_step / 2, centre + self.frame_step / 2

    def frame_range(self, start: float, end: float) -> slice:
        """Frames whose span overlaps [start, end)."""
        first = int(np.floor((start - self.frame_offset) / self.frame_step + 0.5))
        last = int(np.ceil((end - self.frame_offset) / self.frame_step + 0.5))
        return slice(max(0, first), max(0, min(self.frames, last)))

    def turns(self) -> list[tuple[float, float, int]]:
        """Contiguous speech of each speaker: (start, end, speaker), by start."""
        out: list[tuple[float, float, int]] = []
        for k in range(self.num_speakers):
            for first, last in _runs(self.binary[:, k] > 0):
                out.append((self._time(first), self._time(last), k))
        out.sort(key=lambda t: (t[0], t[2]))
        return out

    def overlaps(self) -> list[tuple[float, float]]:
        speaking = self.binary.sum(axis=1)
        return [(self._time(a), self._time(b)) for a, b in _runs(speaking >= 2)]

    def _time(self, frame_edge: int) -> float:
        """Time of the boundary before frame ``frame_edge``."""
        return float(min(self.duration, max(0.0, frame_edge * self.frame_step + self.frame_offset - self.frame_step / 2)))

    def speaker_summary(self) -> list[dict[str, Any]]:
        summary = []
        for k in range(self.num_speakers):
            active = np.nonzero(self.binary[:, k])[0]
            summary.append(
                {
                    "index": k,
                    "speaking_seconds": round(float(active.size * self.frame_step), 3),
                    "first_seconds": round(self._time(int(active[0])), 3) if active.size else None,
                }
            )
        return summary

    def to_rttm(self, uri: str) -> str:
        lines = [
            f"SPEAKER {uri} 1 {start:.3f} {end - start:.3f} <NA> <NA> SPEAKER_{k:02d} <NA> <NA>"
            for start, end, k in self.turns()
            if end > start
        ]
        return "\n".join(lines) + ("\n" if lines else "")

    # persistence (the worker stores the timeline so speaker assignment can be
    # redone without re-running the models)
    def save(self, path: Path) -> None:
        import json

        np.savez_compressed(
            path,
            activation=self.activation,
            count=self.count,
            margin=self.margin,
            binary=self.binary,
            centroids=self.centroids,
            duration=np.array(self.duration),
            info=np.array(json.dumps(self.info)),
        )

    @classmethod
    def load(cls, path: Path) -> "DiarizationResult":
        import json

        with np.load(path, allow_pickle=False) as data:
            return cls(
                duration=float(data["duration"]),
                activation=data["activation"],
                count=data["count"],
                margin=data["margin"],
                binary=data["binary"],
                centroids=data["centroids"],
                info=json.loads(str(data["info"])),
            )


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[first, last+1) index pairs of True runs."""
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


class Diarizer:
    def __init__(self, segmentation_model: Path, embedding_model: Path, *, threads: int, step_seconds: float = DEFAULT_STEP_SECONDS):
        if step_seconds <= 0 or round(step_seconds * SAMPLE_RATE) % FBANK_SHIFT_SAMPLES:
            raise ValueError("window step must be a positive multiple of 10 ms")
        self.segmentation = SegmentationModel(segmentation_model, threads)
        self.embedding = EmbeddingModel(embedding_model, threads)
        self.step_samples = int(round(step_seconds * SAMPLE_RATE))
        self.threads = threads

    def run(
        self,
        reader: WavReader,
        bounds: SpeakerBounds = SpeakerBounds(),
        *,
        progress: ProgressFn | None = None,
        should_cancel: Callable[[], bool] | None = None,
        constrained: bool = False,
    ) -> DiarizationResult:
        started = time.monotonic()
        plan = plan_windows(reader.samples, self.step_samples)
        report = progress or (lambda _f: None)
        timings: dict[str, float] = {}

        # 1. local segmentation
        t0 = time.monotonic()
        activity = segment(
            self.segmentation, reader, plan,
            progress=lambda done: report(_PHASE_WEIGHTS["segmentation"] * done / max(1, plan.count)),
            should_cancel=should_cancel,
        )
        timings["segmentation_seconds"] = time.monotonic() - t0

        # 2. embeddings
        t0 = time.monotonic()
        embeddings, clean = self._embeddings(reader, plan, activity, report, should_cancel)
        timings["embedding_seconds"] = time.monotonic() - t0

        # 3. clustering
        t0 = time.monotonic()
        result = clustering.cluster(embeddings, activity, bounds, constrained=constrained)
        hard = result.hard
        filled = _fill_without_embedding(hard, activity, plan)
        timings["clustering_seconds"] = time.monotonic() - t0

        # 4. timeline
        diar = _reconstruct(activity, hard, result.similarity, plan, reader.samples, bounds)
        diar.centroids = result.centroids[diar.info.pop("order")] if result.centroids.size else result.centroids
        report(1.0)
        active_pairs = int((activity.sum(axis=1) > 0).sum())
        diar.info.update(
            {
                "windows": plan.count,
                "window_seconds": WINDOW_SAMPLES / SAMPLE_RATE,
                "step_seconds": plan.step_seconds,
                "local_speakers": active_pairs,
                "embeddings": int((~np.isnan(embeddings).any(axis=2)).sum()),
                "embeddings_with_overlap": int(((~clean) & ~np.isnan(embeddings).any(axis=2)).sum()),
                "training_embeddings": result.training_count,
                "estimated_speakers": result.estimated,
                "assigned_by_continuity": filled,
                "bounds": {k: v for k, v in bounds.__dict__.items() if v is not None},
                "constrained_assignment": constrained,
                "threshold": clustering.THRESHOLD,
                **{k: round(v, 3) for k, v in timings.items()},
                "total_seconds": round(time.monotonic() - started, 3),
            }
        )
        return diar

    def _embeddings(self, reader: WavReader, plan: WindowPlan, activity: np.ndarray, report: ProgressFn, should_cancel):
        windows = plan.count
        embeddings = np.full((windows, NUM_LOCAL_SPEAKERS, EMBEDDING_DIM), np.nan, dtype=np.float32)
        clean_used = np.zeros((windows, NUM_LOCAL_SPEAKERS), dtype=bool)
        overlap_free = (activity.sum(axis=2, keepdims=True) < 2).astype(np.uint8)
        clean_activity = activity * overlap_free
        batch = 32
        base = _PHASE_WEIGHTS["segmentation"]
        for first in range(0, windows, batch):
            if should_cancel is not None and should_cancel():
                raise Cancelled("The job was cancelled.")
            starts = plan.starts[first : first + batch]
            span_start = int(starts[0])
            span_len = int(starts[-1]) + WINDOW_SAMPLES - span_start
            samples = reader.read_int16(span_start, span_len).astype(np.float32)
            if samples.size < span_len:  # the last window is zero-padded
                samples = np.concatenate([samples, np.zeros(span_len - samples.size, dtype=np.float32)])
            span_feats = filterbank(samples)  # int16-scaled samples, as WeSpeaker expects
            for i, start in enumerate(starts):
                w = first + i
                if not activity[w].any():
                    continue
                feats = window_features(span_feats, (int(start) - span_start) // FBANK_SHIFT_SAMPLES)
                for j in range(NUM_LOCAL_SPEAKERS):
                    mask = activity[w, :, j]
                    if not mask.any():
                        continue
                    # prefer frames where this speaker talks alone (pyannote's
                    # "exclude overlap"); fall back to all of their frames
                    vector = masked_embedding(self.embedding, feats, clean_activity[w, :, j])
                    if vector is not None:
                        clean_used[w, j] = True
                    else:
                        vector = masked_embedding(self.embedding, feats, mask)
                    if vector is not None:
                        embeddings[w, j] = vector
            report(base + _PHASE_WEIGHTS["embeddings"] * min(windows, first + batch) / max(1, windows))
        return embeddings, clean_used


def _fill_without_embedding(hard: np.ndarray, activity: np.ndarray, plan: WindowPlan) -> int:
    """Local speakers with too little speech for an embedding take the global
    speaker of the local speaker they agree with most in overlapping windows
    (their frames coincide). Returns how many were filled."""
    filled = 0
    windows = plan.count
    offsets = np.array([window_frame_offset(int(s)) for s in plan.starts], dtype=np.int64)
    missing = np.argwhere((hard == UNASSIGNED) & (activity.sum(axis=1) > 0))
    for w, j in missing:
        best, best_score = UNASSIGNED, 0
        mine = activity[w, :, j].astype(np.int32)
        for v in range(max(0, w - 12), min(windows, w + 13)):
            if v == w:
                continue
            shift = int(offsets[v] - offsets[w])
            if abs(shift) >= FRAMES_PER_WINDOW:
                continue
            a0, a1 = max(0, shift), min(FRAMES_PER_WINDOW, FRAMES_PER_WINDOW + shift)
            for i in range(NUM_LOCAL_SPEAKERS):
                if hard[v, i] < 0:
                    continue
                theirs = activity[v, a0 - shift : a1 - shift, i]
                score = int(np.minimum(mine[a0:a1], theirs).sum())
                if score > best_score:
                    best, best_score = int(hard[v, i]), score
        if best >= 0:
            hard[w, j] = best
            filled += 1
    return filled


def _reconstruct(
    activity: np.ndarray,
    hard: np.ndarray,
    similarity: np.ndarray,
    plan: WindowPlan,
    total_samples: int,
    bounds: SpeakerBounds,
) -> DiarizationResult:
    """Combine every window's opinion into one global timeline."""
    k = int(hard.max()) + 1 if (hard >= 0).any() else 0
    frames = total_frames(plan)
    real_frames = min(frames, int(np.ceil((total_samples / SAMPLE_RATE) / FRAME_STEP)))
    act_sum = np.zeros((frames, max(k, 1)), dtype=np.float32)
    margin_sum = np.zeros((frames, max(k, 1)), dtype=np.float32)
    margin_weight = np.zeros((frames, max(k, 1)), dtype=np.float32)
    count_sum = np.zeros(frames, dtype=np.float32)
    covered = np.zeros(frames, dtype=np.float32)

    if similarity.size and similarity.shape[2] >= 2:
        ordered = np.sort(np.nan_to_num(similarity, nan=-np.inf), axis=2)
        with np.errstate(invalid="ignore"):  # rows without an embedding stay NaN
            pair_margin = ordered[:, :, -1] - ordered[:, :, -2]
    elif similarity.size:
        pair_margin = np.where(np.isnan(similarity[:, :, 0]), np.nan, 1.0)
    else:
        pair_margin = np.full(hard.shape, np.nan)

    for w in range(plan.count):
        off = window_frame_offset(int(plan.starts[w]))
        sl = slice(off, off + FRAMES_PER_WINDOW)
        covered[sl] += 1
        count_sum[sl] += activity[w].sum(axis=1)
        for g in set(int(x) for x in hard[w] if x >= 0):
            # several local speakers mapped to one global speaker: take the max
            locals_ = [j for j in range(NUM_LOCAL_SPEAKERS) if hard[w, j] == g]
            a = activity[w, :, locals_].max(axis=0).astype(np.float32)
            act_sum[sl, g] += a
            for j in locals_:
                m = pair_margin[w, j]
                if np.isfinite(m):
                    weight = activity[w, :, j].astype(np.float32)
                    margin_sum[sl, g] += weight * float(m)
                    margin_weight[sl, g] += weight

    covered = covered[:real_frames]
    act_sum, count_sum = act_sum[:real_frames], count_sum[:real_frames]
    margin_sum, margin_weight = margin_sum[:real_frames], margin_weight[:real_frames]
    denom = np.maximum(covered, 1.0)
    activation = act_sum / denom[:, None]
    count = np.rint(count_sum / denom).astype(np.int64)
    cap = k
    if bounds.max_speakers:
        cap = min(cap, bounds.max_speakers)
    if bounds.num_speakers:
        cap = min(cap, bounds.num_speakers)
    count = np.minimum(count, cap).astype(np.uint8)
    with np.errstate(invalid="ignore", divide="ignore"):
        margin = np.where(margin_weight > 0, margin_sum / margin_weight, np.nan).astype(np.float32)

    binary = np.zeros_like(activation, dtype=np.uint8)
    if k:
        order = np.argsort(-activation, axis=1, kind="stable")
        for c in range(1, int(count.max(initial=0)) + 1):
            rows = np.nonzero(count >= c)[0]
            chosen = order[rows, c - 1]
            positive = activation[rows, chosen] > 0
            binary[rows[positive], chosen[positive]] = 1
    else:
        activation = activation[:, :0]
        margin = margin[:, :0]
        binary = binary[:, :0]

    # renumber speakers by first appearance; drop speakers that never win a frame
    first_frame = [int(np.argmax(binary[:, g])) if binary[:, g].any() else None for g in range(binary.shape[1])]
    order = [g for g in sorted(range(binary.shape[1]), key=lambda g: (first_frame[g] is None, first_frame[g] or 0)) if first_frame[g] is not None]
    result = DiarizationResult(
        duration=total_samples / SAMPLE_RATE,
        activation=activation[:, order].astype(np.float32),
        count=count,
        margin=margin[:, order].astype(np.float32),
        binary=binary[:, order],
        centroids=np.zeros((0, EMBEDDING_DIM), dtype=np.float32),
        info={"order": order},
    )
    return result
