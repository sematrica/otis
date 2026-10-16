"""Local speaker segmentation with pyannote's segmentation-3.0 (ONNX export).

The model reads 10-second windows of 16 kHz audio and, for each of its 589
output frames (one every 16.875 ms), gives a "powerset" distribution over
seven classes: nobody, one of three local speakers, or one of the three pairs
of local speakers talking at once. Local speaker indices only mean something
inside one window; :mod:`.clustering` links them across windows.

Windows overlap (1 s step by default), so every moment of the recording is
seen by up to ten windows; :mod:`.pipeline` combines their opinions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..audio import SAMPLE_RATE
from ..audio.wav import WavReader
from ..errors import EngineError

WINDOW_SAMPLES = 160_000
FRAMES_PER_WINDOW = 589
#: Output frame step and receptive-field width of segmentation-3.0, from the
#: model's metadata (270 and 991 samples at 16 kHz).
FRAME_STEP = 270 / SAMPLE_RATE
FRAME_DURATION = 991 / SAMPLE_RATE
NUM_LOCAL_SPEAKERS = 3

#: Powerset class -> which of the three local speakers are active.
POWERSET = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ],
    dtype=np.uint8,
)

_EXPECTED_META = {
    "model_type": "pyannote-segmentation-3.0",
    "num_classes": "7",
    "num_speakers": "3",
    "powerset_max_classes": "2",
    "sample_rate": str(SAMPLE_RATE),
    "window_size": str(WINDOW_SAMPLES),
    "receptive_field_shift": "270",
}


@dataclass
class WindowPlan:
    """Start sample of every analysis window; the last one may be zero-padded."""

    starts: np.ndarray
    step_samples: int
    total_samples: int

    @property
    def count(self) -> int:
        return int(self.starts.size)

    @property
    def step_seconds(self) -> float:
        return self.step_samples / SAMPLE_RATE


def plan_windows(total_samples: int, step_samples: int) -> WindowPlan:
    """Same windowing as pyannote's sliding-window inference: regular steps,
    plus one final zero-padded window if the regular ones miss the tail."""
    if total_samples <= 0:
        return WindowPlan(np.zeros(0, dtype=np.int64), step_samples, total_samples)
    if total_samples >= WINDOW_SAMPLES:
        regular = (total_samples - WINDOW_SAMPLES) // step_samples + 1
    else:
        regular = 0
    starts = [i * step_samples for i in range(regular)]
    if total_samples < WINDOW_SAMPLES or (total_samples - WINDOW_SAMPLES) % step_samples > 0:
        starts.append(regular * step_samples)
    return WindowPlan(np.asarray(starts, dtype=np.int64), step_samples, total_samples)


class SegmentationModel:
    def __init__(self, model_path: Path, threads: int):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, threads)
        options.inter_op_num_threads = 1
        try:
            self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        except Exception as exc:  # onnxruntime raises its own exception types
            raise EngineError(f"The speaker segmentation model could not be loaded: {exc}", code="model_invalid") from exc
        meta = self.session.get_modelmeta().custom_metadata_map
        wrong = {k: meta.get(k) for k, v in _EXPECTED_META.items() if meta.get(k) != v}
        if wrong:
            raise EngineError(f"Unexpected speaker segmentation model (metadata {wrong}).", code="model_invalid")
        self._input = self.session.get_inputs()[0].name
        self._output = self.session.get_outputs()[0].name

    def infer(self, windows: np.ndarray) -> np.ndarray:
        """(N, WINDOW_SAMPLES) float32 -> (N, FRAMES_PER_WINDOW, 3) uint8 activity."""
        logits = self.session.run([self._output], {self._input: windows[:, None, :]})[0]
        if logits.shape[1:] != (FRAMES_PER_WINDOW, POWERSET.shape[0]):
            raise EngineError(f"Unexpected segmentation output shape {logits.shape}.", code="model_invalid")
        return POWERSET[np.argmax(logits, axis=-1)]


def segment(
    model: SegmentationModel,
    reader: WavReader,
    plan: WindowPlan,
    *,
    batch_size: int = 32,
    progress: Callable[[int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Run the model over every window. Returns (windows, frames, 3) uint8.

    Audio is read one batch at a time (a batch of 32 windows at a 1 s step
    covers 41 s), so memory does not grow with the recording's length.
    """
    out = np.zeros((plan.count, FRAMES_PER_WINDOW, NUM_LOCAL_SPEAKERS), dtype=np.uint8)
    for first in range(0, plan.count, batch_size):
        if should_cancel is not None and should_cancel():
            from ..errors import Cancelled

            raise Cancelled("The job was cancelled.")
        starts = plan.starts[first : first + batch_size]
        span_start = int(starts[0])
        span = reader.read_int16(span_start, int(starts[-1]) + WINDOW_SAMPLES - span_start).astype(np.float32) / 32768.0
        batch = np.zeros((starts.size, WINDOW_SAMPLES), dtype=np.float32)
        for i, start in enumerate(starts):
            piece = span[int(start) - span_start : int(start) - span_start + WINDOW_SAMPLES]
            batch[i, : piece.size] = piece
        out[first : first + starts.size] = model.infer(batch)
        if progress is not None:
            progress(first + starts.size)
    return out


def window_frame_offset(window_start_sample: int) -> int:
    """Index, on the global frame grid, of a window's first frame."""
    return int(round(window_start_sample / SAMPLE_RATE / FRAME_STEP))


def total_frames(plan: WindowPlan) -> int:
    if plan.count == 0:
        return 0
    return window_frame_offset(int(plan.starts[-1])) + FRAMES_PER_WINDOW


def frame_time(index: np.ndarray | int) -> np.ndarray | float:
    """Centre time (seconds) of global frame(s)."""
    return np.asarray(index) * FRAME_STEP + FRAME_DURATION / 2


def time_to_frame(seconds: float) -> int:
    return max(0, int(math.floor((seconds - FRAME_DURATION / 2) / FRAME_STEP + 0.5)))
