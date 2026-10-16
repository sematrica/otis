"""CTC forced alignment (Viterbi), in NumPy.

Given per-frame log-probabilities from a CTC model and the token sequence of
a known transcript, find the most likely way to emit exactly that sequence:
each token occupies one or more consecutive frames, separated by optional
"blank" frames, in order. This is the algorithm of ``torchaudio.functional.
forced_align`` (which is deprecated) without the PyTorch dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_NEG = np.float32(-1e30)


@dataclass
class TokenSpan:
    target_index: int
    first_frame: int
    last_frame: int  # inclusive
    mean_log_prob: float


@dataclass
class Alignment:
    spans: list[TokenSpan]  # one per target token, in order
    frame_states: np.ndarray  # (T,) target index per frame, -1 for blank
    log_likelihood: float


def min_frames(targets: list[int]) -> int:
    """Frames needed to emit ``targets`` (a repeated token needs a blank between)."""
    repeats = sum(1 for a, b in zip(targets, targets[1:]) if a == b)
    return len(targets) + repeats


def forced_align(log_probs: np.ndarray, targets: list[int], blank: int) -> Alignment | None:
    """Viterbi alignment, or ``None`` if the audio is too short for the text."""
    frames = log_probs.shape[0]
    if not targets:
        return None
    if blank in targets:
        raise ValueError("targets must not contain the blank token")
    if frames < min_frames(targets):
        return None
    ext = np.full(2 * len(targets) + 1, blank, dtype=np.int64)
    ext[1::2] = targets
    states = ext.size
    can_skip = np.zeros(states, dtype=bool)
    can_skip[3::2] = ext[3::2] != ext[1:-2:2]

    lp = log_probs.astype(np.float32, copy=False)
    alpha = np.full(states, _NEG, dtype=np.float32)
    alpha[0] = lp[0, ext[0]]
    alpha[1] = lp[0, ext[1]]
    back = np.zeros((frames, states), dtype=np.int8)
    candidates = np.empty((3, states), dtype=np.float32)
    for t in range(1, frames):
        candidates[0] = alpha
        candidates[1, 0] = _NEG
        candidates[1, 1:] = alpha[:-1]
        candidates[2, :2] = _NEG
        candidates[2, 2:] = np.where(can_skip[2:], alpha[:-2], _NEG)
        choice = np.argmax(candidates, axis=0)
        alpha = candidates[choice, np.arange(states)] + lp[t, ext]
        back[t] = choice

    final = states - 1 if alpha[-1] >= alpha[-2] else states - 2
    total = float(alpha[final])
    if total <= _NEG / 2:
        return None
    path = np.empty(frames, dtype=np.int64)
    s = final
    for t in range(frames - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])
    # states: even = blank, odd = target (s - 1) // 2
    frame_states = np.where(path % 2 == 1, (path - 1) // 2, -1)
    spans: list[TokenSpan] = []
    emitted = lp[np.arange(frames), ext[path]]
    boundaries = np.flatnonzero(np.diff(np.concatenate([[-2], frame_states, [-2]])) != 0)
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        index = int(frame_states[a])
        if index < 0:
            continue
        spans.append(TokenSpan(index, int(a), int(b - 1), float(emitted[a:b].mean())))
    if len(spans) != len(targets):  # cannot happen for a valid path; guard anyway
        return None
    return Alignment(spans, frame_states, total)
