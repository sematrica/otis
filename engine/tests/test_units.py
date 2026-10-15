"""Unit tests that need no model weights."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from verbatim_engine.asr.base import Segment, TranscriptionResult, Word
from verbatim_engine.asr.merge import assemble, normalize_token
from verbatim_engine.audio.chunking import FRAME_SECONDS, Chunk, energy_envelope, plan_chunks
from verbatim_engine.audio.wav import WavReader
from verbatim_engine.export.txt import render_txt
from verbatim_engine.metrics.wer import character_error_rate, normalize, word_error_rate
from verbatim_engine.util.timecode import format_clock, format_precise

from .conftest import SR, write_wav


# ----------------------------------------------------------------------------- corpus integrity


def test_corpus_checksums(corpus):
    from .conftest import CORPUS

    for sample in corpus["samples"]:
        digest = hashlib.sha256((CORPUS / sample["file"]).read_bytes()).hexdigest()
        assert digest == sample["sha256"], sample["id"]


# ----------------------------------------------------------------------------- timecodes & metrics


@pytest.mark.parametrize(
    "seconds, clock, precise",
    [
        (0, "00:00:00", "00:00:00.000"),
        (59.999, "00:00:59", "00:00:59.999"),
        (59.9996, "00:00:59", "00:01:00.000"),
        (72.5, "00:01:12", "00:01:12.500"),
        (3 * 3600 + 5.0004, "03:00:05", "03:00:05.000"),
        (-1, "00:00:00", "00:00:00.000"),
    ],
)
def test_timecodes(seconds, clock, precise):
    assert format_clock(seconds) == clock
    assert format_precise(seconds) == precise


def test_wer_normalisation_and_counts():
    assert normalize("Hello, World!  It's  “fine”.") == "hello world it's fine"
    r = word_error_rate("ask not what your country can do", "ask what your country can do for")
    assert (r.errors, r.deletions, r.insertions, r.substitutions) == (2, 1, 1, 0)
    assert r.reference_length == 7
    assert word_error_rate("a b c", "a b c").rate == 0.0
    assert character_error_rate("abc", "abd").rate == pytest.approx(1 / 3)


# ----------------------------------------------------------------------------- chunk planning


def _bursty_signal(tmp_path, pattern):
    """pattern: list of (seconds, is_speech). Speech = loud noise, else near-silence."""
    rng = np.random.default_rng(0)
    parts = []
    for seconds, loud in pattern:
        n = int(seconds * SR)
        parts.append(rng.normal(0, 0.2 if loud else 0.0005, n).astype(np.float32))
    path = write_wav(tmp_path / "bursty.wav", np.concatenate(parts))
    return path


def test_short_recording_is_single_chunk():
    chunks = plan_chunks(300.0, np.zeros(15000, dtype=np.float32), target_seconds=600)
    assert chunks == [Chunk(0, 0.0, 300.0, 0.0, 300.0, None)]


def test_cuts_land_in_pauses_and_cover_everything(tmp_path):
    # 100 s of speech-like noise with 1 s pauses at 38-39 s and 81-82 s.
    path = _bursty_signal(tmp_path, [(38, True), (1, False), (42, True), (1, False), (18, True)])
    with WavReader(path) as reader:
        env = energy_envelope(reader)
        duration = reader.duration
    assert env.size == pytest.approx(duration / FRAME_SECONDS, abs=1)
    chunks = plan_chunks(duration, env, target_seconds=40, search_seconds=6, pad_seconds=1.0)
    assert len(chunks) == 3
    assert 38.0 < chunks[1].owned_start < 39.0
    assert 81.0 < chunks[2].owned_start < 82.0
    assert all(not c.cut_in_speech for c in chunks)
    # owned intervals tile the recording exactly
    assert chunks[0].owned_start == 0.0 and chunks[-1].owned_end == pytest.approx(duration)
    pauses = [(38.0, 39.0), (81.0, 82.0)]
    for (a, b), (pause_start, pause_end) in zip(zip(chunks, chunks[1:]), pauses):
        assert a.owned_end == b.owned_start
        # audio reaches into the pause but never into the neighbour's speech
        assert a.owned_end <= a.audio_end <= pause_end + FRAME_SECONDS
        assert pause_start - FRAME_SECONDS <= b.audio_start <= b.owned_start
        assert a.audio_end <= a.owned_end + 1.0 and b.audio_start >= b.owned_start - 1.0


def test_cut_without_pause_is_flagged_and_padded(tmp_path):
    path = _bursty_signal(tmp_path, [(100, True)])
    with WavReader(path) as reader:
        env = energy_envelope(reader)
        chunks = plan_chunks(reader.duration, env, target_seconds=40, search_seconds=5, pad_seconds=1.0)
    assert len(chunks) >= 2
    assert all(c.cut_in_speech for c in chunks[1:])
    for a, b in zip(chunks, chunks[1:]):
        assert a.audio_end == pytest.approx(a.owned_end + 1.0)
        assert b.audio_start == pytest.approx(b.owned_start - 1.0)


# ----------------------------------------------------------------------------- merging


def _words(spec):
    """spec: list of (text, start, end)"""
    return [Word(s, e, t, 0.9) for t, s, e in spec]


def _result(segments):
    return TranscriptionResult("en", 0.99, "detected", 100.0, segments)


def test_ownership_drops_words_in_neighbours_region():
    c0 = Chunk(0, 0.0, 10.0, 0.0, 11.0)
    c1 = Chunk(1, 10.0, 20.0, 9.0, 20.0, cut_energy_db=-20.0)  # cut inside speech
    # chunk 0 heard "alpha beta" (before the cut) and "gamma" (after it, in its padding)
    r0 = _result([Segment(5.0, 10.6, " alpha beta gamma", _words([(" alpha", 5.0, 6.0), (" beta", 8.5, 9.5), (" gamma", 10.1, 10.6)]))])
    # chunk 1 (offset 9 s) heard "beta" in its padding and "gamma delta" after the cut
    r1 = _result([Segment(0.0, 4.0, " beta gamma delta", _words([(" beta", -0.5, 0.5), (" gamma", 1.1, 1.6), (" delta", 2.0, 3.0)]))])
    merged = assemble([(c0, r0), (c1, r1)], pad=1.0)
    text = " ".join(s.text for s in merged)
    assert text == "alpha beta gamma delta"
    assert merged[0].flags[0]["type"] == "trimmed_at_chunk_boundary"
    assert merged[1].start == pytest.approx(10.1)


def test_overlap_is_spliced_where_both_chunks_agree():
    c0 = Chunk(0, 0.0, 10.0, 0.0, 11.0)
    c1 = Chunk(1, 10.0, 20.0, 9.0, 20.0, cut_energy_db=-20.0)  # cut inside speech
    # Both chunks heard "we can do it" around the cut, with slightly different timings.
    r0 = _result([Segment(8.0, 10.5, " so we can do it", _words([(" so", 8.0, 8.4), (" we", 9.2, 9.4), (" can", 9.4, 9.6), (" do", 9.8, 10.0), (" it", 10.0, 10.3)]))])
    r1 = _result([Segment(0.4, 4.0, " we can do it. Next topic.", _words([(" we", 0.4, 0.6), (" can", 0.6, 0.8), (" do", 0.95, 1.1), (" it.", 1.1, 1.4), (" Next", 2.5, 2.8), (" topic.", 2.8, 3.2)]))])
    merged = assemble([(c0, r0), (c1, r1)], pad=1.0)
    assert " ".join(s.text for s in merged) == "so we can do it. Next topic."
    boundary = next(f for s in merged for f in s.flags if f["type"] == "chunk_boundary")
    assert boundary["method"] == "aligned"
    assert not any(f["type"] == "boundary_duplicate_removed" for s in merged for f in s.flags)


def test_word_timed_on_opposite_sides_of_cut_is_not_lost():
    # Pure midpoint ownership would drop "maybe" from both chunks: chunk 0 puts
    # it after the cut, chunk 1 before it. Neither chunk agrees on any word, so
    # the loss-free fallback applies.
    c0 = Chunk(0, 0.0, 10.0, 0.0, 11.0)
    c1 = Chunk(1, 10.0, 20.0, 9.0, 20.0, cut_energy_db=-20.0)  # cut inside speech
    r0 = _result([Segment(9.0, 10.4, " right maybe", _words([(" right", 9.0, 9.5), (" maybe", 10.0, 10.4)]))])
    r1 = _result([Segment(0.7, 3.0, " Maybe? Later.", _words([(" Maybe?", 0.7, 0.95), (" Later.", 2.0, 2.6)]))])
    merged = assemble([(c0, r0), (c1, r1)], pad=1.0)
    text = " ".join(s.text for s in merged)
    assert normalize_token(text.split()[1]) == "maybe" and text.endswith("Later.")
    assert sum(1 for t in text.split() if normalize_token(t) == "maybe") == 1


def test_fallback_duplicate_removed_and_recorded():
    c0 = Chunk(0, 0.0, 10.0, 0.0, 11.0)
    c1 = Chunk(1, 10.0, 20.0, 9.0, 20.0, cut_energy_db=-20.0)  # cut inside speech
    # The chunks disagree on timing by 0.7 s (beyond the alignment tolerance for
    # a two-word match), and the copies are only 0.3 s apart: one acoustic event.
    r0 = _result([Segment(8.0, 9.7, " okay do it", _words([(" okay", 8.0, 8.6), (" do", 9.3, 9.5), (" it", 9.5, 9.7)]))])
    r1 = _result([Segment(1.0, 3.0, " do it now", _words([(" do", 1.0, 1.2), (" it", 1.2, 1.4), (" now", 2.5, 3.0)]))])
    merged = assemble([(c0, r0), (c1, r1)], pad=1.0)
    assert " ".join(s.text for s in merged) == "okay do it now"
    flag = next(f for s in merged for f in s.flags if f["type"] == "boundary_duplicate_removed")
    assert flag["text"] == "do it" and flag["duplicate_of_chunk"] == 0


def test_genuine_repetition_across_a_pause_is_kept():
    c0 = Chunk(0, 0.0, 10.0, 0.0, 11.0)
    c1 = Chunk(1, 10.0, 20.0, 9.0, 20.0, cut_energy_db=-20.0)  # cut inside speech
    # "no no", a 0.8 s pause, then "no no" again: a real repetition. Each chunk
    # heard both, as a real decoder would for audio inside its window.
    r0 = _result([Segment(9.0, 11.0, " no no no no", _words([(" no", 9.0, 9.3), (" no", 9.3, 9.6), (" no", 10.4, 10.7), (" no", 10.7, 11.0)]))])
    r1 = _result([Segment(0.0, 2.0, " no no no no", _words([(" no", 0.0, 0.3), (" no", 0.3, 0.6), (" no", 1.4, 1.7), (" no", 1.7, 2.0)]))])
    merged = assemble([(c0, r0), (c1, r1)], pad=1.0)
    assert " ".join(s.text for s in merged) == "no no no no"
    # and when the left chunk missed the second pair entirely, nothing is dropped
    r0_partial = _result([Segment(9.0, 9.6, " no no", _words([(" no", 9.0, 9.3), (" no", 9.3, 9.6)]))])
    r1_late = _result([Segment(1.4, 2.0, " no no", _words([(" no", 1.4, 1.7), (" no", 1.7, 2.0)]))])
    merged = assemble([(c0, r0_partial), (c1, r1_late)], pad=1.0)
    assert " ".join(s.text for s in merged) == "no no no no"
    assert not any(f["type"] == "boundary_duplicate_removed" for s in merged for f in s.flags)


def test_pause_boundary_concatenates_without_changes():
    # Cut in a pause: each chunk's audio stopped at the pause, so nothing was
    # heard twice and even an identical phrase on both sides is kept.
    c0 = Chunk(0, 0.0, 10.0, 0.0, 10.4)
    c1 = Chunk(1, 10.0, 20.0, 9.6, 20.0, cut_energy_db=-80.0)
    r0 = _result([Segment(8.0, 9.5, " I agree.", _words([(" I", 8.0, 8.5), (" agree.", 8.5, 9.5)]))])
    # Whisper stretches the first word back to the start of its window (9.6 s).
    r1 = _result([Segment(0.0, 1.6, " I agree.", _words([(" I", 0.0, 1.0), (" agree.", 1.0, 1.6)]))])
    merged = assemble([(c0, r0), (c1, r1)], pad=1.0)
    assert " ".join(s.text for s in merged) == "I agree. I agree."
    assert merged[1].flags == [{"type": "chunk_boundary", "cut": 10.0, "method": "pause"}]


def test_segments_without_word_timings_are_kept_by_starting_chunk():
    c0 = Chunk(0, 0.0, 10.0, 0.0, 11.0)
    c1 = Chunk(1, 10.0, 20.0, 9.0, 20.0, cut_energy_db=-20.0)  # cut inside speech
    r0 = _result([Segment(2.0, 4.0, " first"), Segment(9.8, 10.8, " straddles late")])
    r1 = _result([Segment(0.5, 1.4, " straddles early"), Segment(3.0, 5.0, " second")])
    merged = assemble([(c0, r0), (c1, r1)], pad=1.0)
    assert [s.text for s in merged] == ["first", "straddles late", "second"]


def test_normalize_token():
    assert normalize_token(" It.") == "it"
    assert normalize_token("¿Qué?") == "qué"


def test_result_round_trip():
    r = _result([Segment(0.0, 1.0, " hi", _words([(" hi", 0.0, 1.0)]), -0.2, 0.01, 1.1, 0.0)])
    again = TranscriptionResult.from_dict(json.loads(json.dumps(r.to_dict())))
    assert again == r


# ----------------------------------------------------------------------------- export


def test_txt_export():
    project = {"title": "Episode 12", "source": {"path": "/x/ep12.mp4"}, "duration": 3725.0}
    transcript = {
        "language": "en",
        "effective": {"backend": "mlx", "model": {"display_name": "Whisper Large-v3"}},
        "segments": [
            {"start": 72.4, "text": "Welcome back to the podcast."},
            {"start": 77.0, "text": " Thank you for having me. "},
            {"start": 80.0, "text": "   "},
        ],
    }
    text = render_txt(project, transcript, {})
    assert text.splitlines() == [
        "Episode 12",
        "Source: ep12.mp4",
        "Duration: 01:02:05",
        "Language: en",
        "Transcribed with: Whisper Large-v3 (mlx)",
        "",
        "[00:01:12] Welcome back to the podcast.",
        "[00:01:17] Thank you for having me.",
    ]
    bare = render_txt(project, transcript, {"timestamps": False, "header": False})
    assert bare == "Welcome back to the podcast.\nThank you for having me.\n"
