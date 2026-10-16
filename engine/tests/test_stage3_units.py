"""Stage 3 unit tests that need no model weights."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from verbatim_engine.alignment.ctc import forced_align
from verbatim_engine.alignment.numbers import cardinal, readings
from verbatim_engine.alignment.score import ctc_log_likelihood

# ----------------------------------------------------------------------------- CTC forward


def _collapse(path: tuple[int, ...], blank: int) -> list[int]:
    out: list[int] = []
    previous = None
    for symbol in path:
        if symbol != previous and symbol != blank:
            out.append(symbol)
        previous = symbol
    return out


def _brute_force(log_probs: np.ndarray, targets: list[int], blank: int) -> float:
    frames, symbols = log_probs.shape
    total = -np.inf
    for path in itertools.product(range(symbols), repeat=frames):
        if _collapse(path, blank) == targets:
            total = np.logaddexp(total, sum(log_probs[t, s] for t, s in enumerate(path)))
    return float(total)


@pytest.mark.parametrize("targets", [[1], [1, 2], [1, 1], [2, 1, 2], []])
def test_forward_algorithm_equals_sum_over_all_paths(targets):
    rng = np.random.default_rng(len(targets) + 7)
    logits = rng.normal(size=(6, 3))
    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    expected = _brute_force(log_probs, targets, blank=0)
    assert ctc_log_likelihood(log_probs, targets, blank=0) == pytest.approx(expected, abs=1e-9)


def test_forward_is_at_least_the_best_path_and_rejects_impossible_text():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(20, 5))
    log_probs = (logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))).astype(np.float32)
    targets = [1, 3, 3, 2]
    viterbi = forced_align(log_probs, targets, blank=0)
    assert viterbi is not None
    assert ctc_log_likelihood(log_probs, targets, blank=0) >= viterbi.log_likelihood - 1e-4
    # a repeated token needs a blank in between: 3 frames cannot emit [1, 1, 1]
    assert ctc_log_likelihood(log_probs[:3], [1, 1, 1], blank=0) is None
    with pytest.raises(ValueError):
        ctc_log_likelihood(log_probs, [0, 1], blank=0)


# ----------------------------------------------------------------------------- number readings


@pytest.mark.parametrize("n, words", [
    (0, "zero"), (7, "seven"), (13, "thirteen"), (40, "forty"), (99, "ninety nine"), (100, "one hundred"),
    (2024, "two thousand twenty four"), (1_000_001, "one million one"),
])
def test_cardinal_numbers(n, words):
    assert " ".join(cardinal(n)) == words


@pytest.mark.parametrize("token, expected", [
    ("1984", ["nineteen eighty four", "one thousand nine hundred eighty four"]),
    ("1905", ["nineteen oh five"]),
    ("5%", ["five percent"]),
    ("$20", ["twenty dollars"]),
    ("$1", ["one dollar"]),
    ("3rd", ["third"]),
    ("21st", ["twenty first"]),
    ("1990s", ["nineteen nineties"]),
    ("3.14", ["three point one four"]),
    ("1,500", ["one thousand five hundred"]),
    ("007", ["zero zero seven"]),
])
def test_number_readings(token, expected):
    found = [" ".join(r) for r in readings(token)]
    for reading in expected:
        assert reading in found, found


def test_non_numbers_have_no_reading():
    assert readings("hello") is None
    assert readings("12abc") is None
    assert readings("") is None


# ----------------------------------------------------------------------------- candidate comparison rules

from verbatim_engine.alignment.score import TextScore  # noqa: E402
from verbatim_engine.verify.compare import DECISION_MARGIN, Candidate, compare  # noqa: E402


class _FixedScorer:
    """Returns preset log-likelihoods per text (anchors included), to exercise
    the decision rules exactly. ``None`` = the model cannot spell the text."""

    def __init__(self, table: dict[str, float | None]):
        self.table = table

    def score(self, _emissions, words):
        value = self.table[" ".join(words)]
        return None if value is None else TextScore(value, len(words))


def _cands(*texts: str) -> list[Candidate]:
    return [Candidate("first_pass" if i == 0 else f"alt{i}", t.split()) for i, t in enumerate(texts)]


def test_first_pass_is_kept_unless_clearly_beaten():
    scorer = _FixedScorer({"a b c": -10.0, "a x c": -10.0 + DECISION_MARGIN - 0.1, "a c": -30.0, "a": -80.0})
    result = compare(scorer, None, _cands("b", "x"), left=["a"], right=["c"])
    assert result.decision == "unresolved" and result.chosen == 0 and result.alternatives == [1]


def test_clear_acoustic_win_replaces_the_first_pass():
    scorer = _FixedScorer({"a b c": -20.0, "a x c": -20.0 + DECISION_MARGIN + 1, "a y c": -40.0, "a c": -60.0})
    result = compare(scorer, None, _cands("b", "x", "y"), left=["a"], right=["c"])
    assert result.decision == "replace" and result.chosen == 1
    assert result.margin == pytest.approx(DECISION_MARGIN + 1)


def test_agreeing_decodes_are_kept_without_alternatives():
    scorer = _FixedScorer({"a b c": -10.0, "a c": -50.0})
    result = compare(scorer, None, _cands("b", "B,", "b"), left=["a"], right=["c"])
    assert result.decision == "keep" and result.alternatives == []  # same words after normalisation


def test_words_without_acoustic_support_are_reported_not_removed():
    scorer = _FixedScorer({"thank you": -40.0, "": -5.0})
    result = compare(scorer, None, _cands("thank you"))
    assert result.decision == "no_acoustic_support" and result.chosen == 0


def test_missing_speech_is_recovered_only_with_clear_evidence():
    clear = compare(_FixedScorer({"": -50.0, "hello there": -20.0}), None, _cands("", "hello there"))
    assert clear.decision == "replace" and clear.chosen == 1
    weak = compare(_FixedScorer({"": -21.0, "hello there": -20.0}), None, _cands("", "hello there"))
    assert weak.decision == "unresolved" and weak.chosen == 0


def test_unspellable_first_pass_cannot_be_judged():
    scorer = _FixedScorer({"≈ b": None, "x b": -5.0, "": -50.0})
    result = compare(scorer, None, _cands("≈ b", "x b"))
    assert result.decision == "unscorable" and result.chosen == 0
