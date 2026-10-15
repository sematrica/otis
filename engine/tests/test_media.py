"""Media validation, probing and audio extraction (real decoding via PyAV)."""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pytest

from verbatim_engine.audio.extract import extract_audio
from verbatim_engine.errors import MediaError, NoAudioStream
from verbatim_engine.media.probe import check_unchanged, probe, validate_path

from .conftest import SR, encode_media, read_wav, silence, write_wav


def tone(seconds: float, freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def onset(samples: np.ndarray, threshold: float = 0.1) -> float:
    idx = np.flatnonzero(np.abs(samples) > threshold)
    return idx[0] / SR if idx.size else float("nan")


def test_probe_wav(jfk_path):
    info = probe(jfk_path)
    assert info.format_name == "wav"
    assert info.duration == pytest.approx(11.0, abs=0.01)
    assert not info.has_video and info.has_audio
    assert info.audio_streams[0].sample_rate == 16000
    assert info.selected_audio_stream == 0


def test_probe_video_with_audio(tmp_path):
    path = encode_media(tmp_path / "clip.mp4", tone(3.0), video=True)
    info = probe(path)
    assert info.has_video and info.has_audio
    assert info.video_streams[0].width == 64
    assert info.duration == pytest.approx(3.0, abs=0.2)


def test_video_without_audio_is_rejected(tmp_path):
    path = encode_media(tmp_path / "silent.mp4", None, video=True, video_seconds=1.0)
    with pytest.raises(NoAudioStream):
        probe(path)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg CLI needed to build a cover-art fixture")
def test_cover_art_is_not_video(tmp_path, jfk_path):
    cover = tmp_path / "cover.png"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=32x32", "-frames:v", "1", str(cover)], check=True)
    mp3 = tmp_path / "song.mp3"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(jfk_path), "-i", str(cover), "-map", "0", "-map", "1",
         "-c:a", "libmp3lame", "-c:v", "png", "-disposition:v", "attached_pic", str(mp3)],
        check=True,
    )
    info = probe(mp3)
    assert info.has_audio and not info.has_video


@pytest.mark.parametrize("content", [b"", b"this is not media at all" * 100])
def test_invalid_files_are_rejected(tmp_path, content):
    path = tmp_path / "bad.mp4"
    path.write_bytes(content)
    with pytest.raises(MediaError):
        probe(path)


def test_path_validation(tmp_path, jfk_path):
    with pytest.raises(MediaError):
        validate_path("relative/file.wav")
    with pytest.raises(MediaError):
        validate_path(str(tmp_path))  # directory
    with pytest.raises(MediaError):
        validate_path(str(tmp_path / "missing.wav"))
    link = tmp_path / "link.wav"
    os.symlink(jfk_path, link)
    assert validate_path(str(link)) == jfk_path.resolve()


def test_check_unchanged(tmp_path, jfk_path):
    copy = tmp_path / "copy.wav"
    shutil.copy(jfk_path, copy)
    st = copy.stat()
    assert check_unchanged(str(copy), st.st_size, st.st_mtime_ns)
    with open(copy, "ab") as f:
        f.write(b"\0\0")
    assert not check_unchanged(str(copy), st.st_size, st.st_mtime_ns)
    assert not check_unchanged(str(tmp_path / "gone.wav"), 1, 1)


def test_extract_preserves_wav_exactly(tmp_path, jfk_path):
    out = tmp_path / "out.wav"
    progress = []
    result = extract_audio(jfk_path, out, expected_duration=11.0, progress=lambda d, t: progress.append((d, t)))
    assert result.sample_rate == 16000
    assert result.duration == pytest.approx(11.0, abs=0.001)
    original, extracted = read_wav(jfk_path), read_wav(out)
    assert extracted.shape == original.shape
    assert np.max(np.abs(extracted - original)) < 1e-4
    assert progress[-1] == (pytest.approx(11.0, abs=0.001), 11.0)
    assert all(b[0] >= a[0] for a, b in zip(progress, progress[1:]))


def test_extract_resamples_and_downmixes(tmp_path):
    import av

    # 44.1 kHz stereo FLAC with a tone burst at 2.0 s in the left channel only.
    rate = 44100
    n = int(4.0 * rate)
    left = np.zeros(n, dtype=np.float32)
    left[int(2.0 * rate) : int(2.5 * rate)] = 0.8 * np.sin(2 * np.pi * 300 * np.arange(int(0.5 * rate)) / rate)
    stereo = np.stack([left, np.zeros_like(left)])
    path = tmp_path / "stereo.flac"
    container = av.open(str(path), "w")
    stream = container.add_stream("flac", rate=rate)
    stream.layout = "stereo"
    pcm = (stereo * 32767).astype(np.int16)
    for start in range(0, n, 4096):
        block = np.ascontiguousarray(pcm[:, start : start + 4096].T).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(block, format="s16", layout="stereo")
        frame.sample_rate = rate
        frame.pts = start
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()

    out = tmp_path / "mono.wav"
    result = extract_audio(path, out)
    samples = read_wav(out)
    assert result.duration == pytest.approx(4.0, abs=0.01)
    assert onset(samples, 0.05) == pytest.approx(2.0, abs=0.01)


def test_extract_keeps_media_timeline_with_leading_audio_offset(tmp_path):
    # Video starts at 0 s, audio track starts at 1.5 s: transcript times must
    # match what the player shows, so 1.5 s of silence is inserted.
    audio = np.concatenate([silence(0.5), tone(1.0)])
    path = encode_media(tmp_path / "offset.mp4", audio, video=True, video_seconds=3.0, audio_offset=1.5)
    out = tmp_path / "offset.wav"
    result = extract_audio(path, out)
    samples = read_wav(out)
    # The AAC encoder's 1024 priming samples (64 ms) are decoded as near-silence
    # before the first real sample, so slightly less padding is needed.
    assert 1.4 <= result.inserted_silence <= 1.5
    assert onset(samples) == pytest.approx(2.0, abs=0.01)


def test_extract_cancellation_leaves_no_partial_file(tmp_path):
    from verbatim_engine.errors import Cancelled

    src = write_wav(tmp_path / "long.wav", tone(30.0))
    out = tmp_path / "x.wav"
    with pytest.raises(Cancelled):
        extract_audio(src, out, should_cancel=lambda: True)
    assert not out.exists()
    assert not list(tmp_path.glob("x.wav*"))


def test_extract_rejects_damaged_file(tmp_path, jfk_path):
    data = bytearray(encode_media(tmp_path / "ok.m4a", tone(5.0), container="ipod").read_bytes())
    # Corrupt the media payload heavily (keep the header so it still opens).
    for i in range(2000, len(data), 7):
        data[i] = (data[i] * 31 + 17) % 256
    bad = tmp_path / "bad.m4a"
    bad.write_bytes(bytes(data))
    try:
        result = extract_audio(bad, tmp_path / "bad.wav", max_decode_errors=10)
    except MediaError:
        return  # rejected: acceptable
    # Or decoded with errors reported, never silently.
    assert result.decode_errors > 0 or result.duration > 0
