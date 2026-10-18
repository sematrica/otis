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


# ----------------------------------------------------------------------------- quality analysis (pass 5)

from verbatim_engine.transcript.model import TSegment, TWord  # noqa: E402
from verbatim_engine.verify.quality import STEP, SpeechMap, analyse, repetitions  # noqa: E402
from verbatim_engine.verify.review import review_priority  # noqa: E402


def _seg(start, words, *, avg=-0.2, prob=0.9, no_speech=0.01, temperature=0.0, compression=1.2, flags=None):
    step = 0.4
    tw = [TWord(start + i * step, start + i * step + 0.3, " " + w, prob) for i, w in enumerate(words)]
    end = tw[-1].end if tw else start + 1.0
    return TSegment(start, end, " ".join(words), tw, "en", avg, no_speech, compression, temperature, 0, list(flags or []))


def _speech(duration, *intervals, diarization=None):
    vad = np.zeros(int(np.ceil(duration / STEP)) + 1, dtype=bool)
    for a, b in intervals:
        vad[int(a / STEP): int(b / STEP)] = True
    return SpeechMap(duration, vad, diarization)


def _types(seg):
    return [f["type"] for f in seg.flags]


def test_repetitions_separate_disfluencies_from_loops():
    assert repetitions("i i think so".split()) == [{"text": "i", "count": 2, "loop": False}]
    assert repetitions("you know you know you know you know right".split())[0] == {
        "text": "you know", "count": 4, "loop": True}
    assert repetitions("thank you thank you".split()) == [{"text": "thank you", "count": 2, "loop": False}]
    assert repetitions("a b c d".split()) == []


def test_clean_segment_with_speech_gets_no_flags():
    seg = _seg(1.0, "this is perfectly clear speech".split())
    assert analyse([seg], _speech(5.0, (0.9, 3.2))) == {}
    assert seg.flags == [] and review_priority(seg.flags) is None


def test_text_over_silence_is_a_possible_hallucination():
    seg = _seg(10.0, "thank you for watching".split())
    analyse([seg], _speech(15.0, (0.0, 2.0)))
    flag = next(f for f in seg.flags if f["type"] == "possible_hallucination")
    assert flag["severity"] == "high"
    assert set(flag["reasons"]) >= {"no_speech_detected", "known_phrase"}


def test_known_phrase_over_real_speech_is_only_a_weak_signal():
    seg = _seg(0.0, "thank you for watching everyone".split())
    analyse([seg], _speech(3.0, (0.0, 3.0)))
    flag = next(f for f in seg.flags if f["type"] == "possible_hallucination")
    assert flag["reasons"] == ["known_phrase"] and flag["severity"] == "low"


def test_unclear_speech_is_flagged_with_its_reasons():
    seg = _seg(0.0, "maybe he said something here".split(), avg=-1.4, prob=0.3, temperature=0.4)
    analyse([seg], _speech(3.0, (0.0, 3.0)))
    flag = next(f for f in seg.flags if f["type"] == "low_confidence")
    assert set(flag["reasons"]) == {"low_average_probability", "many_uncertain_words", "decoder_fallback"}
    assert flag["severity"] == "high" and review_priority(seg.flags) == "high"


def test_a_probability_that_is_not_a_number_is_a_doubt_not_a_value():
    # seen with the MLX decoder in repetition loops: avg_logprob NaN, some word probabilities NaN
    seg = _seg(0.0, "the the the the".split(), avg=float("nan"), prob=0.9)
    seg.words[0].probability = float("nan")
    seg.words[1].probability = float("nan")
    analyse([seg], _speech(3.0, (0.0, 3.0)))
    flag = next(f for f in seg.flags if f["type"] == "low_confidence")
    assert flag["reasons"] == ["probability_unavailable"] and flag["avg_logprob"] is None
    assert flag["words"] == 2 and flag["uncertain_words"] == 0  # the NaN words are not counted as certain
    assert flag["severity"] == "medium"
    low = _seg(5.0, "something unclear".split(), avg=float("-inf"))
    analyse([low], _speech(8.0, (5.0, 6.0)))
    flag = next(f for f in low.flags if f["type"] == "low_confidence")
    assert flag["reasons"] == ["low_average_probability"] and flag["avg_logprob"] is None and flag["severity"] == "high"


def test_loops_and_disfluencies():
    loop = _seg(0.0, "and then and then and then and then".split())
    stutter = _seg(5.0, "i i think so".split())
    analyse([loop, stutter], _speech(8.0, (0.0, 3.2), (5.0, 6.6)))
    assert "possible_hallucination" in _types(loop)
    assert "repetition_loop" in next(f for f in loop.flags if f["type"] == "possible_hallucination")["reasons"]
    assert _types(stutter) == ["repeated_words"] and review_priority(stutter.flags) == "low"


def test_untranscribed_speech_is_reported_once_with_its_detectors():
    first = _seg(0.0, "hello there".split())
    second = _seg(8.0, "goodbye now".split())
    diar = np.zeros(int(np.ceil(10.0 / STEP)) + 1, dtype=bool)
    diar[int(3.0 / STEP): int(6.0 / STEP)] = True
    analyse([first, second], _speech(10.0, (0.0, 0.8), (3.0, 6.0), (8.0, 8.8), diarization=diar))
    gap = next(f for f in first.flags if f["type"] == "missing_speech")
    assert gap["start"] == pytest.approx(3.0, abs=0.05) and gap["end"] == pytest.approx(6.0, abs=0.05)
    assert gap["detectors"] == ["vad", "diarization"] and gap["severity"] == "high"
    assert "missing_speech" not in _types(second)


def test_review_priority_uses_the_most_severe_flag():
    flags = [{"type": "split_at_speaker_change"}, {"type": "speaker_change_uncertain"}]
    assert review_priority(flags) == "low"
    assert review_priority(flags + [{"type": "overlapping_speech"}]) == "medium"
    assert review_priority(flags + [{"type": "marked_uncertain", "by": "user"}]) == "high"
    assert review_priority([{"type": "chunk_boundary"}]) is None


def test_an_empty_decode_never_replaces_words():
    scorer = _FixedScorer({"a b c": -40.0, "a c": -10.0})
    result = compare(scorer, None, _cands("b", ""), left=["a"], right=["c"])
    assert result.chosen == 0 and result.decision == "no_acoustic_support"


# ----------------------------------------------------------------------------- AI review: rules (no model)

from verbatim_engine.errors import InvalidParams  # noqa: E402
from verbatim_engine.llm.providers import OpenAIProvider, is_local, make_provider  # noqa: E402
from verbatim_engine.llm.review import batches, collect, validate  # noqa: E402


@pytest.mark.parametrize("url, local", [
    ("http://127.0.0.1:11434", True), ("http://localhost:11434", True), ("http://[::1]:8080", True),
    ("https://api.openai.com/v1", False), ("http://192.168.1.20:11434", False), ("https://example.org", False),
])
def test_endpoint_locality(url, local):
    assert is_local(url) is local


def test_remote_endpoints_need_https_and_a_key():
    with pytest.raises(InvalidParams):
        make_provider("ollama", "m", base_url="http://192.168.1.20:11434")  # plain http off this computer
    with pytest.raises(InvalidParams):
        make_provider("ollama", "m", base_url="ftp://localhost")
    with pytest.raises(InvalidParams):
        OpenAIProvider("some-model", api_key=None)  # api.openai.com without a key
    assert make_provider("ollama", "m").remote is False
    assert make_provider("openai", "m", api_key="k").remote is True
    assert make_provider("openai", "m", base_url="http://127.0.0.1:11434/v1").remote is False
    with pytest.raises(InvalidParams):
        make_provider("other", "m")


SEGMENTS = {
    1: {"id": 1, "text": " and so my fellow americans ask not", "speaker_id": None},
    2: {"id": 2, "text": " we deployed it on cooper netties last week", "speaker_id": None},
}


def test_punctuation_suggestions_may_not_change_words():
    ok = validate({"segment_id": 1, "kind": "punctuation", "text": "And so, my fellow Americans, ask not", "term": "",
                   "reason": "commas"}, SEGMENTS)
    assert ok["status"] == "pending" and ok["suggested_text"].startswith("And so,")
    changed = validate({"segment_id": 1, "kind": "punctuation", "text": "And so, my fellow citizens, ask not",
                        "term": "", "reason": ""}, SEGMENTS)
    assert changed["status"] == "invalid" and "changes words" in changed["validation"]
    assert validate({"segment_id": 1, "kind": "punctuation", "text": "and so my fellow americans ask not",
                     "term": "", "reason": ""}, SEGMENTS) is None  # no change


def test_terminology_must_name_a_spelling_in_the_passage():
    ok = validate({"segment_id": 2, "kind": "terminology", "text": "Kubernetes", "term": "cooper netties",
                   "reason": "spelled Kubernetes elsewhere"}, SEGMENTS)
    assert ok["status"] == "pending" and ok["suggested_text"] == "we deployed it on Kubernetes last week"
    missing = validate({"segment_id": 2, "kind": "terminology", "text": "Docker", "term": "dokker", "reason": ""},
                       SEGMENTS)
    assert missing["status"] == "invalid"


def test_suggestions_for_unsent_passages_or_unknown_kinds_are_invalid():
    assert validate({"segment_id": 99, "kind": "suspicious", "text": "", "term": "", "reason": "x"},
                    SEGMENTS)["status"] == "invalid"
    assert validate({"segment_id": 1, "kind": "rewrite", "text": "", "term": "", "reason": "x"},
                    SEGMENTS)["status"] == "invalid"
    flagged = validate({"segment_id": 1, "kind": "retranscribe", "text": "", "term": "", "reason": ""}, SEGMENTS)
    assert flagged["status"] == "pending" and flagged["suggested_text"] is None and flagged["reason"]


def test_collect_dedupes_and_survives_malformed_answers():
    batch = list(SEGMENTS.values())
    assert collect({"oops": 1}, batch)[0]["status"] == "invalid"
    item = {"segment_id": 1, "kind": "suspicious", "text": "", "term": "", "reason": "odd"}
    assert len(collect({"suggestions": [item, item, "junk"]}, batch)) == 1


def test_batches_respect_the_word_budget():
    many = [{"id": i, "text": " word" * 300, "speaker_id": None} for i in range(7)]
    out = batches(many, {})
    assert [len(b) for b in out] == [3, 3, 1]


# ----------------------------------------------------------------------------- re-check safeguards

from verbatim_engine.asr.base import Word  # noqa: E402
from verbatim_engine.verify.compare import DELETION_COST  # noqa: E402
from verbatim_engine.verify.retranscribe import Target, extract, select_targets  # noqa: E402


def test_dropping_words_needs_a_lead_per_dropped_word():
    first, shorter = "a b c d", "a d"  # the candidate drops two words
    weak = compare(_FixedScorer({first: -20.0, shorter: -20.0 + DECISION_MARGIN + 1, "": -90.0}), None,
                   _cands(first, shorter))
    assert weak.decision == "unresolved" and weak.chosen == 0
    strong = compare(_FixedScorer({first: -20.0, shorter: -20.0 + DECISION_MARGIN + 2 * DELETION_COST, "": -90.0}),
                     None, _cands(first, shorter))
    assert strong.decision == "replace" and strong.chosen == 1


def test_a_decoded_word_stands_for_one_passage_only():
    context = [TWord(0.0, 0.4, " we"), TWord(0.5, 0.9, " stay"), TWord(1.0, 1.4, " here")]
    decoded = [Word(0.0, 0.4, " we"), Word(0.5, 0.9, " stay"), Word(1.0, 1.4, " near")]
    claimed: set[int] = set()
    first = extract(context, Target(0.9, 1.5, 0), decoded, claimed)
    second = extract(context, Target(0.95, 1.5, 1), decoded, claimed)
    assert [w.text for w in first] == [" near"] and second == []


def test_overlapping_speech_is_not_re_worded():
    a = _seg(0.0, "one two three four five".split(), avg=-1.5, prob=0.2)
    b = _seg(1.0, "six seven eight".split(), avg=-1.5, prob=0.2)  # overlaps a by about a second
    c = _seg(10.0, "nine ten eleven".split(), avg=-1.5, prob=0.2)
    analyse([a, b, c], None)
    skipped: list = []
    targets = select_targets([a, b, c], skipped=skipped)
    assert [t.segment for t in targets] == [2]
    assert [i for i, _ in skipped] == [0, 1] and skipped[0][1] == "overlapping speech"


from verbatim_engine.verify.compare import CandidateScore, Comparison  # noqa: E402
from verbatim_engine.verify.retranscribe import Decision, apply_decisions  # noqa: E402


def _decision(seg: TSegment, decision: str, *texts: str) -> Decision:
    cands = [Candidate("first_pass", [w.text for w in seg.words])] + [
        Candidate(f"alt{i}", [" " + t for t in text.split()]) for i, text in enumerate(texts)]
    timed = [list(seg.words)] + [
        [Word(seg.start + 0.4 * j, seg.start + 0.4 * j + 0.3, " " + t, 0.8) for j, t in enumerate(text.split())]
        for text in texts]
    scores = [CandidateScore(c.source, " ".join(c.words), -10.0) for c in cands]
    comparison = Comparison(decision, 1 if decision == "replace" else 0, 8.0, scores, -50.0,
                            [1] if decision == "unresolved" else [])
    return Decision(Target(seg.start, seg.end, 0), cands, timed, comparison)


def test_checking_a_passage_again_replaces_the_earlier_outcome():
    seg = _seg(0.0, "daddy and a new jersey".split())
    segments, _, _, _ = apply_decisions([seg], [_decision(seg, "unresolved", "diane and new jersey")])
    assert [f["type"] for f in segments[0].flags].count("alternatives") == 1
    segments, _, _, replaced = apply_decisions(segments, [_decision(segments[0], "replace", "diane and new jersey")])
    once = replaced[0]
    assert once.text == "diane and new jersey"
    assert not any(f["type"] in ("alternatives", "rechecked") for f in once.flags)
    segments, _, _, replaced = apply_decisions([once], [_decision(once, "replace", "diane in new jersey")])
    twice = replaced[0]
    retranscribed = [f for f in twice.flags if f["type"] == "retranscribed"]
    assert len(retranscribed) == 1
    assert retranscribed[0]["first_pass"] == "daddy and a new jersey"  # the recogniser's original wording
    again, _, _, _ = apply_decisions([twice], [_decision(twice, "unresolved", "diane on new jersey")])
    again, _, _, _ = apply_decisions(again, [_decision(again[0], "unresolved", "diane on new jersey")])
    types = [f["type"] for f in again[0].flags]
    assert types.count("alternatives") == 1 and types.count("rechecked") == 1


from verbatim_engine.verify.quality import missing_speech  # noqa: E402
from verbatim_engine.verify.validate import validate as validate_transcript  # noqa: E402


def test_an_empty_transcript_over_speech_is_not_reported_as_silence():
    speech = _speech(30.0, (2.0, 2.4), (18.0, 19.5), (28.0, 30.0))
    segments, summary = validate_transcript([], speech, 30.0)
    assert segments == [] and summary["missing_speech"] == 2  # the 0.4 s stretch is below the minimum
    assert summary["untranscribed"][0]["start"] == pytest.approx(18.0, abs=0.05)
    assert summary["untranscribed_seconds"] == pytest.approx(3.5, abs=0.1)
    # Maximum Accuracy re-checks those stretches although no segment carries them
    targets = select_targets([], gaps=missing_speech([], speech))
    assert [(round(t.start), round(t.end), t.gap) for t in targets] == [(18, 20, True), (28, 30, True)]
    assert select_targets([], indices=set(), gaps=missing_speech([], speech)) == []  # not on demand


def test_no_acoustic_support_over_detected_speech_is_a_weaker_doubt():
    for coverage, severity in ((0.0, "high"), (0.9, "medium")):
        seg = _seg(0.0, "oh i dont hear that in new jersey now".split())
        decision = _decision(seg, "no_acoustic_support")
        segments, _, counts, _ = apply_decisions([seg], [decision], speech_coverage=lambda _s, c=coverage: c)
        flag = next(f for f in segments[0].flags if f["type"] == "possible_hallucination")
        assert flag["severity"] == severity and "no_acoustic_support" in flag["reasons"]
        assert segments[0].text == "oh i dont hear that in new jersey now"  # words are never removed
        result = next(f for f in segments[0].flags if f["type"] == "rechecked")
        assert result["speech_detected"] is (coverage > 0.5) and counts["no_acoustic_support"] == 1


def test_a_local_answer_that_never_ends_is_cut_off_and_recorded():
    import http.server
    import json as _json
    import threading

    from verbatim_engine.llm.providers import MAX_ANSWER_TOKENS, LLMError

    seen = []

    class Ollama(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server's naming
            seen.append(_json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = _json.dumps({"message": {"content": '{"suggestions": [' + " " * 50},
                                "done": True, "done_reason": "length"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Ollama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        provider = make_provider("ollama", "small", base_url=f"http://127.0.0.1:{server.server_address[1]}")
        with pytest.raises(LLMError) as raised:
            provider.complete_json("system", "user", {"type": "object"})
    finally:
        server.shutdown()
    assert seen[0]["options"]["num_predict"] == MAX_ANSWER_TOKENS
    assert "cut off" in raised.value.message
    exchange = raised.value.data["exchange"]
    assert exchange["response"].startswith('{"suggestions": [') and "cut off" in exchange["error"]
