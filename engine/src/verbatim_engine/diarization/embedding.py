"""Speaker embeddings with WeSpeaker ResNet34 (VoxCeleb, ONNX export).

Same model and feature recipe as pyannote 3.1's ``wespeaker-voxceleb-resnet34-LM``:
80-bin Kaldi filterbanks (25 ms Hamming frames every 10 ms, no dither) on
int16-scaled samples, mean-normalised over each 10 s window, then only the
frames where a given local speaker talks (alone, when possible) go into the
network, which returns one 256-dimensional embedding per local speaker.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..audio import SAMPLE_RATE
from ..errors import EngineError
from .segmentation import FRAMES_PER_WINDOW, WINDOW_SAMPLES

FBANK_SHIFT_SAMPLES = 160  # 10 ms
FBANK_LENGTH_SAMPLES = 400  # 25 ms
FBANK_FRAMES_PER_WINDOW = (WINDOW_SAMPLES - FBANK_LENGTH_SAMPLES) // FBANK_SHIFT_SAMPLES + 1  # 998
EMBEDDING_DIM = 256
#: Fewer frames than this (0.2 s of speech) give no usable speaker evidence.
MIN_EMBEDDING_FRAMES = 20

#: Nearest-neighbour map from filterbank frames to segmentation frames
#: (what pyannote does with ``F.interpolate(mode="nearest")``).
_FBANK_TO_SEG = np.floor(np.arange(FBANK_FRAMES_PER_WINDOW) * FRAMES_PER_WINDOW / FBANK_FRAMES_PER_WINDOW).astype(np.int64)


def filterbank(samples_int16_scale: np.ndarray) -> np.ndarray:
    """Kaldi-compatible log-mel filterbank: (frames, 80) float32."""
    import kaldi_native_fbank as knf

    options = knf.FbankOptions()
    options.frame_opts.samp_freq = SAMPLE_RATE
    options.frame_opts.frame_length_ms = 25
    options.frame_opts.frame_shift_ms = 10
    options.frame_opts.dither = 0.0
    options.frame_opts.window_type = "hamming"
    options.frame_opts.snip_edges = True
    options.mel_opts.num_bins = 80
    options.use_energy = False
    computer = knf.OnlineFbank(options)
    computer.accept_waveform(SAMPLE_RATE, samples_int16_scale.astype(np.float32, copy=False))
    computer.input_finished()
    count = computer.num_frames_ready
    if count == 0:
        return np.zeros((0, 80), dtype=np.float32)
    return np.stack([computer.get_frame(i) for i in range(count)]).astype(np.float32, copy=False)


class EmbeddingModel:
    def __init__(self, model_path: Path, threads: int):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, threads)
        options.inter_op_num_threads = 1
        try:
            self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise EngineError(f"The speaker embedding model could not be loaded: {exc}", code="model_invalid") from exc
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape[-1] != 80 or outputs[0].shape[-1] != EMBEDDING_DIM:
            raise EngineError("Unexpected speaker embedding model.", code="model_invalid")
        self._input = inputs[0].name
        self._output = outputs[0].name

    def embed(self, features: np.ndarray) -> np.ndarray:
        """(frames, 80) -> (256,) float32."""
        return self.session.run([self._output], {self._input: features[None]})[0][0].astype(np.float32)


def window_features(span_features: np.ndarray, offset_frames: int) -> np.ndarray:
    """Mean-normalised filterbanks for one 10 s window cut from a longer span."""
    feats = span_features[offset_frames : offset_frames + FBANK_FRAMES_PER_WINDOW]
    if feats.shape[0] < FBANK_FRAMES_PER_WINDOW:
        raise ValueError("span too short for window")
    return feats - feats.mean(axis=0, keepdims=True)


def masked_embedding(model: EmbeddingModel, feats: np.ndarray, seg_mask: np.ndarray) -> np.ndarray | None:
    """Embedding of the frames where ``seg_mask`` (segmentation frames) is set."""
    selected = feats[seg_mask[_FBANK_TO_SEG] > 0]
    if selected.shape[0] < MIN_EMBEDDING_FRAMES:
        return None
    return model.embed(np.ascontiguousarray(selected))
