"""Stage 4 YouTube import, without YouTube (it is not reachable from the
development container; CI tries the real service separately).

Real yt-dlp runs against a local web page with an HTML5 video and a caption
track (its generic extractor), through the same retrieval code; the URL rules,
caption parsing, format choice and refusals are checked directly. The format
lists in ``test_format_choice`` have the shape yt-dlp reports for YouTube
(itags 140/251 audio, 136/248 video, 18 combined) — they describe the input,
they are not recorded output.
"""

from __future__ import annotations

import functools
import http.server
import threading

import pytest

from verbatim_engine.errors import InvalidParams
from verbatim_engine.youtube import captions, fetch, urls
from verbatim_engine.youtube.urls import TEST_HOST_VARIABLE

from tests.conftest import encode_media, read_wav


@pytest.mark.parametrize("text, video_id", [
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("youtube.com/watch?v=dQw4w9WgXcQ&t=42s&list=PL123", "dQw4w9WgXcQ"),
    ("http://m.youtube.com/watch?feature=share&v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://music.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://youtu.be/dQw4w9WgXcQ?si=abc", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
])
def test_video_links_become_one_canonical_url(text, video_id):
    target = urls.validate(text)
    assert target.video_id == video_id and target.url == f"https://www.youtube.com/watch?v={video_id}"


@pytest.mark.parametrize("text", [
    "", "not a url", "ftp://youtube.com/watch?v=dQw4w9WgXcQ", "file:///etc/passwd",
    "https://vimeo.com/123456", "https://youtube.com.evil.example/watch?v=dQw4w9WgXcQ",
    "https://evil.example/?u=https://youtube.com/watch?v=dQw4w9WgXcQ",
    "https://user:pw@www.youtube.com/watch?v=dQw4w9WgXcQ", "https://www.youtube.com:8443/watch?v=dQw4w9WgXcQ",
    "https://www.youtube.com/playlist?list=PL123", "https://www.youtube.com/@channel",
    "https://www.youtube.com/watch?v=short", "https://www.youtube.com/watch?v=dQw4w9WgXcQ\nx",
    "http://127.0.0.1:8000/index.html",  # loopback pages only when a test names that host
])
def test_everything_else_is_refused_before_any_network_access(text):
    with pytest.raises(InvalidParams):
        urls.validate(text)


def test_the_test_host_can_only_be_loopback(monkeypatch):
    monkeypatch.setenv(TEST_HOST_VARIABLE, "evil.example:80")
    with pytest.raises(InvalidParams):
        urls.validate("http://evil.example:80/index.html")
    monkeypatch.setenv(TEST_HOST_VARIABLE, "127.0.0.1:8123")
    assert urls.validate("http://127.0.0.1:8123/index.html").video_id is None


CREATOR_VTT = """WEBVTT
Kind: captions
Language: en

00:00:01.000 --> 00:00:03.500
And so, my fellow Americans,

00:00:03.500 --> 00:00:06.000 align:start position:0%
ask not what your country
<i>can do for you</i> &amp; me
"""

# YouTube's automatic captions: each cue repeats the previous line, the new
# words carry inline timestamps, and 10 ms cues hold the finished line.
ROLLING_VTT = """WEBVTT
Kind: captions
Language: en

00:00:00.320 --> 00:00:02.310 align:start position:0%
\x20
and<00:00:00.640><c> so</c><00:00:00.960><c> my</c><00:00:01.280><c> fellow</c>

00:00:02.310 --> 00:00:02.320 align:start position:0%
and so my fellow
\x20

00:00:02.320 --> 00:00:04.990 align:start position:0%
and so my fellow
americans<00:00:02.960><c> ask</c><00:00:03.200><c> not</c>

00:00:04.990 --> 00:00:05.000 align:start position:0%
americans ask not
\x20
"""


def test_creator_and_rolling_captions_are_read_without_repeats():
    creator = captions.parse_vtt(CREATOR_VTT)
    assert [c.text for c in creator] == ["And so, my fellow Americans,", "ask not what your country can do for you & me"]
    assert (creator[0].start, creator[1].end) == (1.0, 6.0)
    rolling = captions.parse_vtt(ROLLING_VTT)
    assert [c.text for c in rolling] == ["and so my fellow", "americans ask not"]
    assert " ".join(c.text for c in rolling).split() == "and so my fellow americans ask not".split()


def test_clear_disagreement_with_captions_is_flagged_softly():
    cues = captions.parse_vtt(CREATOR_VTT)
    agree = {"start": 1.0, "end": 3.4, "text": "And so my fellow Americans"}
    differ = {"start": 3.5, "end": 6.0, "text": "a snot what you're cutting candy food"}
    flags = captions.compare([agree, differ], cues, kind="manual", language="en")
    assert list(flags) == [1]
    # caption words up to 0.3 s outside a passage count as its own
    assert flags[1]["severity"] == "low" and "ask not what your country" in flags[1]["caption_text"]
    assert captions.same_language("en-orig", "en") and not captions.same_language("de", "en")


def _fmt(fid, ext, vcodec, acodec, height=None, abr=None, tbr=None, size=None, drm=False):
    return {"format_id": fid, "ext": ext, "vcodec": vcodec, "acodec": acodec, "height": height, "abr": abr,
            "tbr": tbr, "filesize": size, "url": f"https://example.invalid/{fid}", "has_drm": drm}


def test_format_choice():
    info = {"formats": [
        _fmt("251", "webm", "none", "opus", abr=135, size=4_000_000),
        _fmt("140", "m4a", "none", "mp4a.40.2", abr=129, size=3_800_000),
        _fmt("136", "mp4", "avc1.4d401f", "none", height=720, tbr=1200, size=30_000_000),
        _fmt("248", "webm", "vp9", "none", height=1080, tbr=2500, size=60_000_000),
        _fmt("18", "mp4", "avc1.42001E", "mp4a.40.2", height=360, tbr=500, size=12_000_000),
    ]}
    both = fetch.choose_formats(info, video=True)
    assert (both.audio["format_id"], both.video["format_id"], both.combined) == ("140", "136", None)
    assert both.approx_bytes == 33_800_000
    audio = fetch.choose_formats(info, video=False)
    assert audio.audio["format_id"] == "140" and audio.video is None
    no_separate_video = {"formats": [f for f in info["formats"] if f["format_id"] in ("140", "18")]}
    assert fetch.choose_formats(no_separate_video, video=True).combined["format_id"] == "18"
    with pytest.raises(fetch.YouTubeError) as drm:
        fetch.choose_formats({"formats": [_fmt("140", "m4a", "none", "mp4a.40.2", drm=True)]}, video=False)
    assert drm.value.code == "youtube_drm"
    with pytest.raises(fetch.YouTubeError) as nothing:
        fetch.choose_formats({"formats": [_fmt("137", "mp4", "avc1", "none", height=1080)]}, video=False)
    assert nothing.value.code == "youtube_unsupported"  # no audio, and no DRM either
    # a page's <audio> element: no codec information, no video
    page_audio = {"formats": [{"format_id": "0", "ext": "wav", "vcodec": "none", "url": "http://127.0.0.1/a.wav"}]}
    assert fetch.choose_formats(page_audio, video=False).audio["format_id"] == "0"
    assert fetch.choose_formats(page_audio, video=True).audio["format_id"] == "0"  # no video to add


def test_refusals_and_sponsor_chapters():
    assert fetch.refusal({"live_status": "is_live", "formats": []})["code"] == "youtube_live"
    assert fetch.refusal({"availability": "subscriber_only", "formats": []})["code"] == "youtube_restricted"
    assert fetch.refusal({"formats": [_fmt("1", "mp4", "avc1", "mp4a", drm=True)]})["code"] == "youtube_drm"
    assert fetch.refusal({"live_status": "was_live", "availability": "public", "formats": []}) is None
    chapters = {"chapters": [{"start_time": 0, "end_time": 60, "title": "Intro"},
                             {"start_time": 60, "end_time": 120, "title": "Sponsor: SomeVPN"},
                             {"start_time": 120, "end_time": 300, "title": "Adding numbers"}]}
    assert [c["title"] for c in fetch.sponsor_chapters(chapters)] == ["Sponsor: SomeVPN"]


def test_errors_are_explained():
    assert fetch._explain(Exception("Sign in to confirm you’re not a bot")).code == "youtube_blocked"
    assert fetch._explain(Exception("Private video. Sign in if you've been granted access")).code == "youtube_restricted"
    assert fetch._explain(Exception("Video unavailable")).code == "youtube_unavailable"
    assert fetch._explain(Exception("The uploader has not made this video available in your country")).code \
        == "youtube_restricted"


@pytest.fixture()
def media_site(tmp_path, jfk_path, monkeypatch):
    """A local page with an HTML5 video (JFK with a test picture) and captions."""
    site = tmp_path / "site"
    site.mkdir()
    encode_media(site / "talk.mp4", read_wav(jfk_path), video=True)
    (site / "talk.en.vtt").write_text(CREATOR_VTT)
    (site / "index.html").write_text(
        '<html><head><title>JFK talk</title></head><body><video controls><source src="talk.mp4" type="video/mp4">'
        '<track kind="captions" src="talk.en.vtt" srclang="en" label="English"></video></body></html>')
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host = f"127.0.0.1:{server.server_address[1]}"
    monkeypatch.setenv(TEST_HOST_VARIABLE, host)
    yield f"http://{host}/index.html"
    server.shutdown()


def test_real_retrieval_of_media_and_captions(media_site, tmp_path):
    import av

    target = urls.validate(media_site)
    inspected = fetch.inspect(target)
    assert inspected["refusal"] is None and inspected["title"].startswith("JFK talk")
    assert [(c["language"], c["kind"], c["formats"]) for c in inspected["captions"]] == [("en", "manual", ["vtt"])]
    info = fetch._extract(target, {"skip_download": True})
    folder = tmp_path / "media"
    seen = []
    got = fetch.retrieve(target, info, folder, video=True, progress=lambda d, t: seen.append((d, t)))
    assert got.path.exists() and got.bytes == got.path.stat().st_size and got.has_video
    with av.open(str(got.path)) as container:
        assert container.streams.audio and container.streams.video
    again = fetch.retrieve(target, info, folder, video=True)  # an earlier attempt's file is reused
    assert again.path == got.path
    tracks = fetch.retrieve_captions(target, inspected["captions"], folder)
    assert tracks[0]["kind"] == "manual" and tracks[0]["cues"][0]["text"] == "And so, my fellow Americans,"


def test_cancelling_a_download(media_site, tmp_path):
    target = urls.validate(media_site)
    info = fetch._extract(target, {"skip_download": True})
    with pytest.raises(fetch.Cancelled):
        fetch.retrieve(target, info, tmp_path / "media", video=True, should_cancel=lambda: True)


def test_remux_keeps_both_streams_without_re_encoding(tmp_path, jfk_path):
    import av

    audio = encode_media(tmp_path / "audio.m4a", read_wav(jfk_path), container="mp4")
    video = encode_media(tmp_path / "video.mp4", None, video=True, video_seconds=11.0)
    out = fetch.remux(video, audio, tmp_path / "media.mp4")
    with av.open(str(out)) as merged, av.open(str(audio)) as source:
        assert merged.streams.video[0].codec_context.name == "mpeg4"
        assert merged.streams.audio[0].codec_context.name == source.streams.audio[0].codec_context.name
        assert abs(float(merged.duration / av.time_base) - 11.0) < 0.5
