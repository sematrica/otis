"""Stage 2 logic that needs no model weights: CTC alignment, boundary
refinement, level normalisation, clustering, DER, speaker assignment,
storage migration, corrections with undo/redo, exports, options and
fixed-URL model downloads."""

from __future__ import annotations

import hashlib
import http.server
import io
import json
import sqlite3
import tarfile
import threading
from pathlib import Path

import numpy as np
import pytest

from verbatim_engine.alignment import refine
from verbatim_engine.alignment.aligner import WordAligner, plan_groups
from verbatim_engine.alignment.ctc import forced_align, min_frames
from verbatim_engine.audio import SAMPLE_RATE
from verbatim_engine.audio.level import even_out
from verbatim_engine.diarization import clustering
from verbatim_engine.diarization.assign import assign_speakers
from verbatim_engine.diarization.pipeline import DiarizationResult
from verbatim_engine.diarization.segmentation import FRAME_DURATION, FRAME_STEP, plan_windows
from verbatim_engine.errors import DownloadError, InvalidJobState, InvalidParams, ModelNotInstalled, NotFound
from verbatim_engine.export.txt import render_txt
from verbatim_engine.jobs.options import normalize_options, resolve
from verbatim_engine.media.probe import probe
from verbatim_engine.metrics.der import diarization_error_rate, parse_rttm
from verbatim_engine.models import catalog
from verbatim_engine.models.catalog import ModelSpec, RemoteFile
from verbatim_engine.models.manager import MANIFEST_NAME, ModelManager
from verbatim_engine.paths import EnginePaths
from verbatim_engine.storage import db as dbmod
from verbatim_engine.storage import repo
from verbatim_engine.storage.db import Database
from verbatim_engine.transcript.build import apply_alignment, speakers_in_order
from verbatim_engine.transcript.edit import TranscriptEditor, rederive_words
from verbatim_engine.transcript.model import TSegment, TWord

# ----------------------------------------------------------------------------- CTC alignment


def _emissions(frames: int, vocab: int, peaks: dict[int, int], blank: int = 0) -> np.ndarray:
    """Log-probabilities: blank everywhere except the given frame -> token peaks."""
    probs = np.full((frames, vocab), 0.01)
    probs[:, blank] = 0.9
    for frame, token in peaks.items():
        probs[frame] = 0.01
        probs[frame, token] = 0.9
    probs /= probs.sum(axis=1, keepdims=True)
    return np.log(probs).astype(np.float32)


def test_forced_align_follows_the_emissions():
    lp = _emissions(20, 5, {3: 1, 4: 2, 10: 3, 15: 1})
    alignment = forced_align(lp, [1, 2, 3, 1], blank=0)
    assert [(s.first_frame, s.last_frame) for s in alignment.spans] == [(3, 3), (4, 4), (10, 10), (15, 15)]
    assert all(s.mean_log_prob > np.log(0.5) for s in alignment.spans)


def test_forced_align_needs_a_blank_between_repeats_and_enough_frames():
    assert min_frames([1, 1, 2]) == 4
    assert forced_align(_emissions(3, 4, {}), [1, 1, 2], blank=0) is None
    alignment = forced_align(_emissions(6, 4, {1: 1, 3: 1, 4: 2}), [1, 1, 2], blank=0)
    assert alignment is not None
    first, second = alignment.spans[0], alignment.spans[1]
    assert first.last_frame < second.first_frame - 0  # separated
    assert alignment.frame_states[first.last_frame + 1] == -1 or second.first_frame > first.last_frame + 1
    with pytest.raises(ValueError):
        forced_align(_emissions(5, 4, {}), [0, 1], blank=0)


class _FakeModel:
    """A CTC 'model' whose emissions are produced from known word positions."""

    frame_seconds = 0.02
    blank = 0
    vocab_size = 4
    id = "fake"
    display_name = "fake"

    def __init__(self, peaks_seconds: list[float]):
        self.peaks = peaks_seconds

    def supports(self, language):
        return True

    def emissions(self, audio):
        frames = int(np.ceil(audio.size / SAMPLE_RATE / self.frame_seconds))
        offset = self._offset
        peaks = {int(round((t - offset) / self.frame_seconds)): 1 for t in self.peaks
                 if 0 <= (t - offset) / self.frame_seconds < frames}
        return _emissions(frames, self.vocab_size, peaks)

    def tokenize(self, word, language):
        return None if word.strip().isdigit() else [1]


class _Reader:
    def __init__(self, duration: float):
        self.duration = duration
        self.samples = int(duration * SAMPLE_RATE)

    def read(self, start, end):
        # silence: boundary refinement then has no sound to extend words over
        return np.zeros(int((end - start) * SAMPLE_RATE), np.float32)


class _W:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


class _S:
    def __init__(self, start, end, words):
        self.start, self.end, self.words = start, end, words


def test_aligner_moves_words_to_the_acoustic_evidence_and_times_wildcards():
    model = _FakeModel([1.0, 1.5, 3.0])
    segments = [_S(0.8, 3.5, [_W(0.8, 1.1, " one"), _W(1.1, 1.9, " two"), _W(1.9, 2.5, " 42"), _W(2.5, 3.4, " three")])]
    aligner = WordAligner(model)
    group = plan_groups(segments)[0]
    start = max(0.0, segments[0].start - 0.6)
    model._offset = start
    result = aligner.align(_Reader(5.0), segments, "en")[0]
    assert result.status == "aligned"
    timings = result.words
    assert [t.source for t in timings] == ["alignment", "alignment", "wildcard", "alignment"]
    assert abs(timings[0].start - 1.0) < 0.03 and abs(timings[3].start - 3.0) < 0.03
    # in connected speech a word starts between the previous word's evidence and its own
    assert 1.0 < timings[1].start <= 1.5 and timings[0].end == timings[1].start
    # the number is timed by the wildcard between its neighbours
    assert timings[1].end <= timings[2].start + 1e-6 and timings[2].end <= timings[3].start + 1e-6
    assert group == [0]


def test_aligner_rejects_text_that_does_not_match():
    model = _FakeModel([])  # the audio shows no speech at all
    model._offset = 0.0
    segments = [_S(0.0, 2.0, [_W(0.0, 1.0, " hello"), _W(1.0, 2.0, " world")])]
    result = WordAligner(model).align(_Reader(3.0), segments, "en")[0]
    assert result.status == "failed" and "match" in result.reason


def test_plan_groups_respects_gaps_and_length():
    segs = [_S(0, 5, [_W(0, 1, "a")]), _S(5.5, 10, [_W(6, 7, "b")]), _S(20, 25, [_W(20, 21, "c")]), _S(25, 60, [_W(26, 27, "d")])]
    assert plan_groups(segs) == [[0, 1], [2], [3]]


# ----------------------------------------------------------------------------- boundary refinement


def _tone(seconds: float, level: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (level * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_refine_uses_pauses_and_splits_connected_gaps():
    # sound 0.5-1.5 s, pause 1.5-2.0, sound 2.0-3.0 s
    audio = np.concatenate([np.zeros(8000, np.float32), _tone(1.0), np.zeros(8000, np.float32), _tone(1.0),
                            np.zeros(8000, np.float32)])
    spans = [refine.Span(0.6, 1.2), refine.Span(2.1, 2.8)]
    out = refine.refine(spans, audio, 0.0)
    assert abs(out[0].end - 1.5) < 0.02 and abs(out[1].start - 2.0) < 0.02
    assert abs(out[0].start - 0.5) < 0.02  # first word extended back over its sound (bounded)
    assert abs(out[1].end - 3.0) < 0.02
    # connected: no pause between the spikes -> split at CONNECTED_SPLIT of the gap
    connected = refine.refine([refine.Span(0.6, 1.0), refine.Span(1.2, 1.4)], _tone(2.0), 0.0)
    assert connected[0].end == pytest.approx(1.0 + refine.CONNECTED_SPLIT * 0.2)
    assert connected[1].start == connected[0].end


def test_even_out_raises_quiet_speech_and_leaves_silence():
    quiet = _tone(1.0, level=0.01)
    loud = _tone(1.0, level=0.3)
    silence = np.zeros(SAMPLE_RATE, np.float32)
    audio = np.concatenate([loud, silence, quiet])
    out = even_out(audio)
    assert np.abs(out[:SAMPLE_RATE]).max() == pytest.approx(np.abs(loud).max(), rel=0.05)  # not attenuated
    assert np.abs(out[SAMPLE_RATE + 2000 : 2 * SAMPLE_RATE - 2000]).max() == 0.0
    gain = np.abs(out[-SAMPLE_RATE // 2 :]).max() / np.abs(quiet).max()
    assert gain > 4.0  # raised by more than 12 dB


# ----------------------------------------------------------------------------- clustering and DER


def _cluster_data(seed: int = 0, per: int = 30, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((k, 256))
    embeddings = np.full((per * k, 3, 256), np.nan, dtype=np.float32)
    activity = np.zeros((per * k, 589, 3), dtype=np.uint8)
    for i in range(per * k):
        embeddings[i, 0] = centres[i % k] + 0.15 * rng.standard_normal(256)
        activity[i, :400, 0] = 1
    return embeddings, activity


def test_clustering_estimates_the_number_of_speakers():
    embeddings, activity = _cluster_data()
    result = clustering.cluster(embeddings, activity)
    assert result.estimated == 3 and result.centroids.shape[0] == 3
    labels = result.hard[:, 0]
    assert len(set(labels[0::3])) == 1 and len(set(labels[1::3])) == 1 and len(set(labels)) == 3
    assert (result.hard[:, 1:] == clustering.UNASSIGNED).all()


def test_clustering_honours_speaker_bounds():
    embeddings, activity = _cluster_data()
    two = clustering.cluster(embeddings, activity, clustering.SpeakerBounds(num_speakers=2))
    assert two.centroids.shape[0] == 2
    one = clustering.cluster(embeddings, activity, clustering.SpeakerBounds(max_speakers=1))
    assert one.centroids.shape[0] == 1 and set(one.hard[:, 0]) == {0}
    assert clustering.SpeakerBounds(min_speakers=4, max_speakers=2).resolve(10) == (2, 2, 2)


def test_der_metric():
    ref = [(0.0, 10.0, "A"), (10.0, 20.0, "B")]
    assert diarization_error_rate(ref, [(0.0, 10.0, "x"), (10.0, 20.0, "y")]).der == pytest.approx(0.0)
    swapped = diarization_error_rate(ref, [(0.0, 10.0, "y"), (10.0, 20.0, "x")])
    assert swapped.der == pytest.approx(0.0) and swapped.mapping == {"y": "A", "x": "B"}
    one = diarization_error_rate(ref, [(0.0, 20.0, "x")])
    assert one.confusion == pytest.approx(10.0, abs=0.02) and one.der == pytest.approx(0.5, abs=0.002)
    missed = diarization_error_rate(ref, [(0.0, 10.0, "x")])
    assert missed.missed == pytest.approx(10.0, abs=0.02)
    overlap = diarization_error_rate([(0, 10, "A"), (5, 10, "B")], [(0, 10, "x")])
    assert overlap.total == pytest.approx(15.0, abs=0.02) and overlap.missed == pytest.approx(5.0, abs=0.02)
    collar = diarization_error_rate(ref, [(0.0, 10.2, "x"), (10.2, 20.0, "y")], collar=0.25)
    assert collar.der == pytest.approx(0.0)
    assert parse_rttm("SPEAKER f 1 1.5 2.0 <NA> <NA> spk <NA> <NA>\n") == {"f": [(1.5, 3.5, "spk")]}


def test_window_plan_matches_pyannote_sliding_windows():
    plan = plan_windows(30 * SAMPLE_RATE, SAMPLE_RATE)
    assert plan.count == 21 and plan.starts[-1] == 20 * SAMPLE_RATE
    tail = plan_windows(30 * SAMPLE_RATE + 1, SAMPLE_RATE)
    assert tail.count == 22  # one zero-padded window for the extra sample
    short = plan_windows(3 * SAMPLE_RATE, SAMPLE_RATE)
    assert short.count == 1 and short.starts[0] == 0


# ----------------------------------------------------------------------------- speaker assignment


def _diarization(intervals: list[tuple[float, float, int]], duration: float, k: int) -> DiarizationResult:
    frames = int(np.ceil(duration / FRAME_STEP))
    binary = np.zeros((frames, k), dtype=np.uint8)
    centres = np.arange(frames) * FRAME_STEP + FRAME_DURATION / 2
    for start, end, speaker in intervals:
        binary[(centres >= start) & (centres < end), speaker] = 1
    return DiarizationResult(
        duration=duration, activation=binary.astype(np.float32), count=binary.sum(axis=1).astype(np.uint8),
        margin=np.where(binary > 0, 0.5, np.nan).astype(np.float32), binary=binary,
        centroids=np.zeros((k, 256), np.float32),
    )


def _segment(words: list[tuple[float, float, str]]) -> TSegment:
    ws = [TWord(a, b, t, 0.9) for a, b, t in words]
    return TSegment(words[0][0], words[-1][1], "".join(w.text for w in ws), ws, language="en",
                    flags=[{"type": "chunk_boundary", "method": "pause"}])


def test_assignment_splits_segments_at_speaker_changes():
    diar = _diarization([(0.0, 2.0, 0), (2.0, 4.0, 1)], 5.0, 2)
    seg = _segment([(0.1, 0.5, " Thank"), (0.5, 1.0, " you."), (2.1, 2.5, " My"), (2.5, 3.0, " pleasure.")])
    out = assign_speakers([seg], diar)
    assert [s.text for s in out] == [" Thank you.", " My pleasure."]
    assert [s.speaker for s in out] == [0, 1]
    assert all(w.speaker == 0 for w in out[0].words) and all(w.speaker == 1 for w in out[1].words)
    assert out[0].start == seg.start and out[1].end == seg.end and out[1].start == 2.1
    assert [f["type"] for f in out[0].flags] == ["chunk_boundary", "split_at_speaker_change"]
    assert [f["type"] for f in out[1].flags] == ["split_at_speaker_change"]
    assert "".join(s.text for s in out) == seg.text  # no word lost or altered
    assert out[0].speaker_confidence == pytest.approx(1.0)


def test_assignment_flags_overlap_uncertainty_and_missing_speech():
    diar = _diarization([(0.0, 3.0, 0), (1.0, 3.0, 1), (6.0, 7.0, 1)], 12.0, 2)
    seg = _segment([(1.2, 1.8, " Both"), (1.8, 2.6, " talking.")])
    out = assign_speakers([seg], diar)[0]
    overlap = next(f for f in out.flags if f["type"] == "overlapping_speech")
    assert overlap["fraction"] == pytest.approx(1.0) and overlap["other_speakers"] == [1 - out.speaker]
    near = assign_speakers([_segment([(7.3, 7.6, " later")])], diar)[0]
    assert near.speaker == 1
    assert "no_speech_under_words" in next(f for f in near.flags if f["type"] == "speaker_uncertain")["reasons"]
    lost = assign_speakers([_segment([(10.0, 10.5, " nobody")])], diar)[0]
    assert lost.speaker is None and lost.flags[-1]["type"] == "speaker_unassigned"


def test_assignment_folds_weak_switches_inside_a_turn():
    # speaker 1 barely touches the middle word: it stays with speaker 0
    diar = _diarization([(0.0, 3.0, 0), (1.45, 1.5, 1)], 4.0, 2)
    diar.binary[:, 0][(np.arange(diar.frames) * FRAME_STEP + FRAME_DURATION / 2 >= 1.4) &
                      (np.arange(diar.frames) * FRAME_STEP + FRAME_DURATION / 2 < 1.6)] = 0
    seg = _segment([(0.2, 0.9, " one"), (1.2, 1.9, " two"), (2.0, 2.8, " three")])
    out = assign_speakers([seg], diar)
    assert len(out) == 1 and out[0].speaker == 0


def test_speakers_are_numbered_by_first_appearance():
    segs = [TSegment(5, 6, "b", speaker=3), TSegment(1, 2, "a", speaker=7), TSegment(7, 8, "c", speaker=3)]
    assert speakers_in_order(segs) == [{"index": 7, "label": "Speaker 1", "color_index": 0},
                                       {"index": 3, "label": "Speaker 2", "color_index": 1}]


def test_apply_alignment_keeps_whisper_times_and_flags_failures():
    from verbatim_engine.alignment.aligner import SegmentAlignment, WordTiming

    a = _segment([(0.0, 1.0, " a"), (1.0, 2.0, " b")])
    for w in a.words:
        w.whisper_start, w.whisper_end = w.start, w.end
    b = _segment([(3.0, 4.0, " c")])
    summary = apply_alignment([a, b], [
        SegmentAlignment("aligned", None, [WordTiming(0.2, 0.8, -0.1, "alignment"), WordTiming(0.9, 1.7, None, "wildcard")]),
        SegmentAlignment("failed", "the audio does not match the text well enough", None, -6.0),
    ])
    assert summary == {"segments": 2, "aligned": 1, "failed": 1, "skipped": 0, "wildcard_words": 1}
    assert (a.start, a.end) == (0.2, 1.7) and a.words[0].whisper_start == 0.0 and a.words[1].timing_source == "wildcard"
    assert b.words[0].start == 3.0 and b.flags[-1]["type"] == "alignment_failed"


# ----------------------------------------------------------------------------- storage and corrections


@pytest.fixture()
def store(data_dir, jfk_path):
    paths = EnginePaths.create(data_dir)
    db = Database(paths.db_path)
    project = repo.create_project(db, probe(jfk_path), "Interview")
    job = repo.create_job(db, project["id"], "balanced", normalize_options({}))
    repo.finish_job(db, job["id"], "completed")

    def seg(start, end, words, speaker):
        ws = [TWord(a, b, t, 0.9, "alignment", a, b, -0.1, speaker) for a, b, t in words]
        return TSegment(start, end, "".join(w.text for w in ws), ws, language="en", speaker=speaker, speaker_confidence=0.9)

    segments = [
        seg(0, 2, [(0, 0.5, " Hello"), (0.5, 1, ","), (1, 2, " there.")], 0),
        seg(2.5, 4, [(2.5, 3, " Hi"), (3, 4, " Bob.")], 1),
        seg(4.5, 6, [(4.5, 5, " How"), (5, 5.5, " are"), (5.5, 6, " you?")], 0),
    ]
    segments[1].flags = [{"type": "overlapping_speech", "other_speakers": [0]}]
    speakers = [{"index": 0, "label": "Speaker 1", "color_index": 0}, {"index": 1, "label": "Speaker 2", "color_index": 1}]
    repo.replace_transcript(db, project["id"], job["id"], segments, "en", speakers)
    yield db, project["id"], TranscriptEditor(db)
    db.close()


def _state(db, project_id):
    t = repo.get_transcript(db, project_id, include_words=True)
    return [(s["text"], s["speaker_id"], [(w["text"], w["start"], w["speaker_id"]) for w in s["words"]]) for s in t["segments"]], \
        [(s["id"], s["display_name"]) for s in t["speakers"]]


def test_transcript_payload_has_speakers_words_and_original(store):
    db, pid, _ = store
    t = repo.get_transcript(db, pid, include_words=True)
    one, two = t["speakers"]
    assert (one["label"], one["segment_count"], two["label"], two["segment_count"]) == ("Speaker 1", 2, "Speaker 2", 1)
    assert t["segments"][1]["flags"][0]["other_speakers"] == [one["id"]]  # diarization index -> speaker id
    assert t["segments"][0]["words"][0]["speaker_id"] == one["id"]
    assert t["history"]["can_undo"] is False and t["revision"] >= 1
    original = repo.get_original_transcript(db, pid)
    assert [s["text"] for s in original["segments"]] == [" Hello, there.", " Hi Bob.", " How are you?"]


def test_corrections_undo_and_redo_restore_every_state(store):
    db, pid, editor = store
    states = [_state(db, pid)]
    speakers = {name: sid for sid, name in states[0][1]}
    seg_ids = [s["id"] for s in repo.get_transcript(db, pid)["segments"]]

    editor.rename_speaker(pid, speakers["Speaker 1"], "  Alice ")
    states.append(_state(db, pid))
    assert ("Alice" in [n for _, n in states[-1][1]])
    editor.split_segment(pid, seg_ids[0], 2)
    states.append(_state(db, pid))
    assert [s[0] for s in states[-1][0]][:2] == [" Hello,", " there."]
    second = repo.get_transcript(db, pid)["segments"][1]["id"]
    editor.edit_text(pid, second, "there my friend.")
    states.append(_state(db, pid))
    words = repo.get_transcript(db, pid, include_words=True)["segments"][1]["words"]
    assert [w["text"] for w in words] == [" there", " my", " friend."]
    assert words[0]["timing_source"] == "alignment" and words[1]["timing_source"] == "interpolated"
    editor.merge_speakers(pid, [speakers["Speaker 2"]], speakers["Speaker 1"])
    states.append(_state(db, pid))
    t = repo.get_transcript(db, pid)
    assert [s["display_name"] for s in t["speakers"]] == ["Alice"] and len(t["merged_speakers"]) == 1
    assert {s["speaker_id"] for s in t["segments"]} == {speakers["Speaker 1"]}
    assert t["segments"][2]["original_speaker_id"] == speakers["Speaker 2"]  # the AI's choice is kept
    _, bob = editor.assign_speaker(pid, [t["segments"][2]["id"]], new_speaker_name="Bob")
    states.append(_state(db, pid))
    t = repo.get_transcript(db, pid)
    assert t["segments"][2]["speaker_id"] == bob and t["segments"][2]["speaker_edited"] is True
    editor.merge_segments(pid, [t["segments"][0]["id"], t["segments"][1]["id"]])
    states.append(_state(db, pid))
    assert repo.get_transcript(db, pid)["segments"][0]["text"] == " Hello, there my friend."
    editor.mark_uncertain(pid, repo.get_transcript(db, pid)["segments"][0]["id"], True, "check name")
    states.append(_state(db, pid))

    # undo everything, checking each intermediate state, then redo everything
    for expected in reversed(states[:-1]):
        assert editor.undo(pid) is not None
        assert _state(db, pid) == expected
    assert editor.undo(pid) is None
    for expected in states[1:]:
        assert editor.redo(pid) is not None
        assert _state(db, pid) == expected
    assert editor.redo(pid) is None
    # the AI original never changed
    assert [s["text"] for s in repo.get_original_transcript(db, pid)["segments"]] == [" Hello, there.", " Hi Bob.", " How are you?"]
    assert len(editor.history(pid)) == len(states) - 1


def test_new_edit_discards_redo_history(store):
    db, pid, editor = store
    sid = repo.get_transcript(db, pid)["speakers"][0]["id"]
    editor.rename_speaker(pid, sid, "A")
    editor.undo(pid)
    assert repo.get_transcript(db, pid)["history"]["can_redo"] is True
    editor.rename_speaker(pid, sid, "B")
    history = repo.get_transcript(db, pid)["history"]
    assert history["can_redo"] is False and history["undo_summary"] == "Rename “Speaker 1” to “B”"


def test_corrections_validate_input_and_running_jobs(store):
    db, pid, editor = store
    t = repo.get_transcript(db, pid)
    seg_ids = [s["id"] for s in t["segments"]]
    with pytest.raises(InvalidParams):
        editor.split_segment(pid, seg_ids[1], 0)
    with pytest.raises(InvalidParams):
        editor.merge_segments(pid, [seg_ids[0], seg_ids[2]])  # not adjacent
    with pytest.raises(InvalidParams):
        editor.edit_text(pid, seg_ids[0], "   ")
    with pytest.raises(NotFound):
        editor.rename_speaker(pid, 9999, "x")
    with pytest.raises(InvalidParams):
        editor.merge_speakers(pid, [t["speakers"][0]["id"]], t["speakers"][0]["id"])
    assert editor.rename_speaker(pid, t["speakers"][0]["id"], None) is None  # no change, no history entry
    repo.create_job(db, pid, "fast", normalize_options({"mode": "fast"}))  # queued
    with pytest.raises(InvalidJobState):
        editor.rename_speaker(pid, t["speakers"][0]["id"], "Alice")


def test_rederive_words_keeps_timings_of_unchanged_words():
    old = [
        {"start": 0.0, "end": 0.4, "text": " the", "probability": 0.9, "timing_source": "alignment",
         "whisper_start": 0.0, "whisper_end": 0.4, "alignment_score": -0.1, "original_text": " the"},
        {"start": 0.4, "end": 0.9, "text": " cat", "probability": 0.8, "timing_source": "alignment",
         "whisper_start": 0.4, "whisper_end": 0.9, "alignment_score": -0.2, "original_text": " cat"},
        {"start": 1.0, "end": 1.4, "text": " sat.", "probability": 0.9, "timing_source": "alignment",
         "whisper_start": 1.0, "whisper_end": 1.4, "alignment_score": -0.1, "original_text": " sat."},
    ]
    words = rederive_words(old, " The black cat sat down.", 0.0, 2.0)
    assert [w["text"] for w in words] == [" The", " black", " cat", " sat", " down."]
    assert words[0]["start"] == 0.0 and words[0]["original_text"] == " the"  # case change keeps timing
    assert words[1]["timing_source"] == "interpolated" and 0.4 <= words[1]["start"] <= 0.4
    assert words[2]["start"] == 0.4 and words[3]["start"] == 1.0
    assert words[4]["timing_source"] == "interpolated"


def test_txt_export_groups_speaker_turns():
    project = {"title": "Episode", "source": {"path": "/x/e.mp4"}, "duration": 100.0}
    transcript = {
        "language": "en",
        "speakers": [{"id": 1, "display_name": "Speaker 1"}, {"id": 2, "display_name": "Ana"}],
        "segments": [
            {"start": 72.4, "speaker_id": 1, "text": " Welcome back to the podcast."},
            {"start": 75.0, "speaker_id": 1, "text": " Today we talk about owls."},
            {"start": 77.0, "speaker_id": 2, "text": " Thank you for having me."},
            {"start": 80.0, "speaker_id": 1, "text": " Let's begin."},
        ],
    }
    lines = render_txt(project, transcript, {"header": False}).splitlines()
    assert lines == ["[00:01:12] Speaker 1", "Welcome back to the podcast. Today we talk about owls.", "",
                     "[00:01:17] Ana", "Thank you for having me.", "", "[00:01:20] Speaker 1", "Let's begin."]


def test_schema_v2_migrates_a_stage1_database(tmp_path):
    path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON")
    for statement in dbmod._split_sql(dbmod.SCHEMA_V1):
        conn.execute(statement)
    conn.execute("PRAGMA user_version = 1")
    conn.execute("INSERT INTO projects (id, title, created_at, updated_at, status, media_info_json) VALUES ('p','t','x','x','ready','{}')")
    conn.execute("INSERT INTO segments (id, project_id, ordinal, start, end, original_text, text) VALUES (7,'p',0,0,1,' a',' a')")
    conn.execute("INSERT INTO words (segment_id, ordinal, start, end, text) VALUES (7, 0, 0, 0.5, ' a')")
    conn.commit()
    conn.close()
    db = Database(path)
    assert db.schema_version == dbmod.LATEST_VERSION
    seg = db.query_one("SELECT * FROM segments WHERE id = 7")
    assert seg["text"] == " a" and seg["speaker_id"] is None and seg["text_edited"] == 0
    word = db.query_one("SELECT * FROM words WHERE segment_id = 7")
    assert word["original_text"] == " a" and word["timing_source"] == "whisper"
    assert db.query_one("PRAGMA foreign_keys")[0] == 1
    db.execute("DELETE FROM segments WHERE id = 7")
    assert db.query_one("SELECT COUNT(*) FROM words")[0] == 0  # cascade still works after the rebuild
    db.close()


def test_old_jobs_get_new_stages_when_they_start(store):
    db, pid, _ = store
    job = repo.create_job(db, pid, "fast", normalize_options({"mode": "fast"}))
    db.execute("DELETE FROM job_stages WHERE job_id = ? AND stage IN ('diarize', 'align')", (job["id"],))
    repo.start_stage(db, job["id"], "diarize", detail="x")
    stages = [s["stage"] for s in repo.get_job(db, job["id"])["stages"]]
    assert "diarize" in stages and stages.index("diarize") > stages.index("transcribe")


# ----------------------------------------------------------------------------- options


def test_speaker_and_alignment_options():
    assert normalize_options({})["diarize"] == "auto" and normalize_options({})["align"] == "auto"
    assert normalize_options({"mode": "fast"})["align"] is False
    opts = normalize_options({"diarize": True, "speakers": {"min": 2, "max": 4}})
    assert opts["speakers"] == {"num": None, "min": 2, "max": 4}
    for bad in ({"speakers": {"num": 0}}, {"speakers": {"num": 2, "min": 1}}, {"speakers": {"min": 5, "max": 2}},
                {"speakers": {"other": 1}}, {"diarize": "yes"}, {"alignment_model": "faster-whisper/small"}):
        with pytest.raises((InvalidParams, NotFound)):
            normalize_options(bad)


def _install(models: ModelManager, spec: ModelSpec, files: dict[str, bytes]) -> None:
    path = models.install_path(spec)
    path.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (path / name).write_bytes(data)
    (path / MANIFEST_NAME).write_text(json.dumps({"model_id": spec.id}))


def test_resolve_speakers_and_alignment(data_dir):
    paths = EnginePaths.create(data_dir)
    db = Database(paths.db_path)
    models = ModelManager(db, paths)
    whisper = catalog.get_spec("faster-whisper/small")
    _install(models, whisper, {"model.bin": b"x", "config.json": b"{}", "tokenizer.json": b"{}"})
    options = normalize_options({"mode": "fast", "backend": "faster-whisper", "align": "auto"})
    resolved = resolve(options, models)
    assert resolved.diarization_dir is None and "not installed" in resolved.diarization_skip_reason
    assert resolved.alignment_candidates == []
    with pytest.raises(ModelNotInstalled) as err:
        resolve({**options, "diarize": True}, models)
    assert err.value.data["model_id"] == "diarization/pyannote-wespeaker"
    with pytest.raises(ModelNotInstalled):
        resolve({**options, "align": True}, models)

    _install(models, catalog.get_spec("diarization/pyannote-wespeaker"), {"segmentation.onnx": b"x", "embedding.onnx": b"x"})
    _install(models, catalog.get_spec("alignment/en-conformer-small"), {"model.onnx": b"x", "tokens.txt": b"x"})
    _install(models, catalog.get_spec("alignment/mms-300m"), {"aligner.onnx": b"x"})
    resolved = resolve({**options, "speakers": {"num": 2, "min": None, "max": None}}, models)
    assert resolved.diarization_dir is not None and resolved.speaker_bounds["num"] == 2
    # the multilingual model measured most precise; it is only present if the user installed it
    assert [s.id for s, _ in resolved.alignment_candidates] == ["alignment/mms-300m", "alignment/en-conformer-small"]
    assert resolved.alignment_model_for("en")[0].id == "alignment/mms-300m"
    assert resolved.alignment_model_for("de")[0].id == "alignment/mms-300m"
    german = resolve({**options, "language": "de"}, models)
    assert [s.id for s, _ in german.alignment_candidates] == ["alignment/mms-300m"]
    assert resolve({**options, "diarize": False, "align": False}, models).describe()["diarization"]["enabled"] is False
    db.close()


# ----------------------------------------------------------------------------- downloads


class _Server:
    def __init__(self, directory: Path):
        handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(directory), **k)  # noqa: E731
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self) -> None:
        self.httpd.shutdown()


def test_fixed_url_download_verifies_checksums_and_extracts(data_dir, tmp_path, monkeypatch):
    serve = tmp_path / "srv"
    serve.mkdir()
    payload = b"onnx-model-bytes" * 1000
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:bz2") as tar:
        for name, data in (("pkg/model.onnx", payload), ("pkg/tokens.txt", b"a 0\n"), ("pkg/extra.bin", b"junk")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    (serve / "pkg.tar.bz2").write_bytes(buffer.getvalue())
    (serve / "single.onnx").write_bytes(payload)
    server = _Server(serve)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    try:
        archive = RemoteFile(f"{server.base}/pkg.tar.bz2", hashlib.sha256(buffer.getvalue()).hexdigest(),
                             len(buffer.getvalue()), members=(("pkg/model.onnx", "model.onnx"), ("pkg/tokens.txt", "tokens.txt")))
        single = RemoteFile(f"{server.base}/single.onnx", hashlib.sha256(payload).hexdigest(), len(payload), name="extra.onnx")
        spec = ModelSpec(id="alignment/test", backend="onnx", family="test", display_name="Test", repo_id=None,
                         parameters="1", approx_download_bytes=archive.size + single.size, kind="alignment",
                         files=(archive, single), required_files=("model.onnx", "tokens.txt", "extra.onnx"), license="test")
        monkeypatch.setitem(catalog._BY_ID, spec.id, spec)
        paths = EnginePaths.create(data_dir)
        db = Database(paths.db_path)
        models = ModelManager(db, paths)
        seen = []
        status = models.download(spec.id, progress=lambda done, total: seen.append((done, total)))
        assert status["installed"] and seen[-1] == (spec.approx_download_bytes, spec.approx_download_bytes)
        installed = Path(status["path"])
        assert (installed / "model.onnx").read_bytes() == payload
        assert sorted(p.name for p in installed.iterdir()) == ["extra.onnx", "model.onnx", "tokens.txt", MANIFEST_NAME]

        corrupt = RemoteFile(single.url, "0" * 64, single.size, name="bad.onnx")
        bad = ModelSpec(**{**spec.__dict__, "id": "alignment/bad", "family": "bad", "files": (corrupt,),
                           "required_files": ("bad.onnx",)})
        monkeypatch.setitem(catalog._BY_ID, bad.id, bad)
        with pytest.raises(DownloadError, match="checksum"):
            models.download(bad.id)
        assert not models.status(bad)["installed"]
        db.close()
    finally:
        server.close()


def test_catalog_entries_are_consistent():
    for spec in catalog.ALL_MODELS:
        if spec.kind != "asr":
            assert spec.license and spec.required_files
            assert spec.files or spec.repo_id
            if spec.files:
                assert spec.approx_download_bytes == sum(f.size for f in spec.files)
                assert all(len(f.sha256) == 64 and f.url.startswith("https://") for f in spec.files)
    assert "non-commercial" in catalog.get_spec("alignment/mms-300m").license
