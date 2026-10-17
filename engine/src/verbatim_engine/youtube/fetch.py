"""Metadata, media and caption retrieval with yt-dlp, within YouTube's rules.

What this module does *not* do: sign in (no cookies, no netrc, no browser
profiles), bypass geographic restrictions (``geo_bypass`` is off), download
DRM-protected or unplayable formats, follow links to other sites (only the
YouTube extractor may run), or fetch code at run time (yt-dlp's JavaScript
components come from the pinned ``yt-dlp-ejs`` package and run in the pinned
``deno``). Live streams and videos that need an account are refused with an
explanation.

Media is retrieved as YouTube serves it: an AAC audio stream and, unless the
user wants audio only, an H.264 video stream up to 720p, downloaded
separately (resumable) and put into one MP4 by PyAV without re-encoding, so
the app can play it. Everything goes into one folder owned by the project.
"""

from __future__ import annotations

import heapq
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..errors import EngineError
from .captions import Cue, parse_vtt
from .urls import MediaURL

log = logging.getLogger(__name__)

MAX_VIDEO_HEIGHT = 720
#: chapter titles that mark a sponsor segment or ad inside the video itself
SPONSOR_CHAPTER = re.compile(r"\b(sponsor(ed|ship)?|ad(vert(isement)?)?s?|werbung|anuncio|publicidad|publicit[ée])\b",
                             re.IGNORECASE)
ADS_NOTE = ("Transcribed from the media file retrieved from YouTube. Ads that YouTube inserts while a video plays are "
            "not part of that file and are not transcribed. Sponsor messages or ads that are part of the video itself "
            "are transcribed like any other speech.")
#: refusal reasons from the metadata
_LIVE = {"is_live": "This is a live stream. Import it after it has ended and YouTube has processed it.",
         "is_upcoming": "This live stream or premiere has not started yet.",
         "post_live": "This live stream has just ended and YouTube is still processing it. Try again later."}
_NEEDS_ACCOUNT = {"private", "premium_only", "subscriber_only", "needs_auth"}


class YouTubeError(EngineError):
    code = "youtube_error"


class Cancelled(Exception):
    """Raised from a progress hook to stop a download."""


class _Log:
    def __init__(self):
        self.warnings: list[str] = []

    def debug(self, message: str) -> None:
        if message.startswith("[debug] "):
            return
        log.debug("yt-dlp: %s", message)

    def info(self, message: str) -> None:
        log.debug("yt-dlp: %s", message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)
        log.info("yt-dlp warning: %s", message)

    def error(self, message: str) -> None:
        log.info("yt-dlp error: %s", message)


def _deno() -> str | None:
    try:
        import deno

        path = deno.find_deno_bin()
        return path if path and os.access(path, os.X_OK) else None
    except Exception:  # noqa: BLE001 - the package may be missing in a development setup
        return shutil.which("deno")


def base_options(target: MediaURL, logger: _Log) -> dict[str, Any]:
    runtime = _deno()
    options: dict[str, Any] = {
        "quiet": True, "no_warnings": False, "noprogress": True, "logger": logger,
        "noplaylist": True, "lazy_playlist": True, "playlistend": 1,
        # no sign-in of any kind, no restriction bypass, nothing unplayable
        "cookiefile": None, "cookiesfrombrowser": None, "usenetrc": False, "username": None, "password": None,
        "geo_bypass": False, "allow_unplayable_formats": False, "remote_components": [],
        "nocheckcertificate": False, "socket_timeout": 30, "retries": 5, "fragment_retries": 5,
        "continuedl": True, "overwrites": False, "restrictfilenames": True, "noresizebuffer": False,
        "allowed_extractors": ["youtube"] if target.is_youtube else ["generic", "html5"],
        "cachedir": False,
    }
    if runtime:
        options["js_runtimes"] = {"deno": {"path": runtime}}
    return options


def _explain(error: Exception) -> YouTubeError:
    """yt-dlp's message turned into what the user can do about it."""
    text = str(error)
    lowered = text.lower()
    if "not a bot" in lowered or "confirm you" in lowered and "bot" in lowered:
        return YouTubeError("YouTube refused the request from this computer and asks to sign in to prove it is not "
                            "a bot. Verbatim does not sign in to YouTube. Try again later, or download the video "
                            "where you are allowed to and import the file.", code="youtube_blocked")
    if any(k in lowered for k in ("private video", "members-only", "member", "sign in to confirm your age",
                                  "age-restricted", "requires payment", "premium", "login required", "sign in")):
        return YouTubeError("This video needs a YouTube account (private, members-only, age-restricted or paid). "
                            "Verbatim does not sign in or bypass restrictions.", code="youtube_restricted")
    if "drm" in lowered:
        return YouTubeError("This video is protected by DRM and cannot be imported.", code="youtube_drm")
    if "available in your country" in lowered or "geo" in lowered and "restrict" in lowered:
        return YouTubeError("This video is not available in your country. Verbatim does not bypass geographic "
                            "restrictions.", code="youtube_restricted")
    if any(k in lowered for k in ("video unavailable", "has been removed", "does not exist", "is not available")):
        return YouTubeError("YouTube says this video is unavailable.", code="youtube_unavailable")
    if any(k in lowered for k in ("timed out", "getaddrinfo", "name or service", "connection", "network",
                                  "urlopen error", "temporary failure")):
        return YouTubeError("YouTube could not be reached. Check the internet connection and try again.",
                            code="youtube_network")
    return YouTubeError(f"YouTube import failed: {text.removeprefix('ERROR: ')}", code="youtube_error")


def _extract(target: MediaURL, extra: dict[str, Any] | None = None, *, download: bool = False) -> dict[str, Any]:
    import yt_dlp

    logger = _Log()
    options = {**base_options(target, logger), **(extra or {})}
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(target.url, download=download)
            return ydl.sanitize_info(info) or {}
    except Cancelled:
        raise
    except yt_dlp.utils.DownloadError as exc:
        if isinstance(exc.exc_info[1] if exc.exc_info else None, Cancelled):
            raise Cancelled() from exc
        raise _explain(exc) from exc
    except yt_dlp.utils.YoutubeDLError as exc:
        raise _explain(exc) from exc


# ----------------------------------------------------------------------------- metadata


@dataclass
class Choice:
    """The formats to retrieve: one combined file, or audio (+ video)."""
    audio: dict[str, Any] | None
    video: dict[str, Any] | None
    combined: dict[str, Any] | None

    @property
    def approx_bytes(self) -> int | None:
        parts = [f for f in (self.combined,) if f] or [f for f in (self.audio, self.video) if f]
        sizes = [f.get("filesize") or f.get("filesize_approx") for f in parts]
        return int(sum(sizes)) if sizes and all(sizes) else None


def _codec(f: dict[str, Any], key: str) -> str:
    return (f.get(key) or "none").lower()


def choose_formats(info: dict[str, Any], *, video: bool) -> Choice:
    formats = [f for f in info.get("formats") or [] if not f.get("has_drm") and f.get("url")]
    audio_only = [f for f in formats if _codec(f, "vcodec") == "none" and _codec(f, "acodec") != "none"]
    aac = [f for f in audio_only if _codec(f, "acodec").startswith("mp4a") or f.get("ext") == "m4a"]
    best_audio = max(aac or audio_only, key=lambda f: (f.get("abr") or f.get("tbr") or 0), default=None)
    video_only = [f for f in formats if _codec(f, "acodec") == "none" and _codec(f, "vcodec") != "none"
                  and (f.get("height") or 0) <= MAX_VIDEO_HEIGHT]
    h264 = [f for f in video_only if _codec(f, "vcodec").startswith("avc1") and f.get("ext") == "mp4"]
    best_video = max(h264, key=lambda f: (f.get("height") or 0, f.get("tbr") or 0), default=None)
    combined = [f for f in formats if _codec(f, "vcodec") != "none" and _codec(f, "acodec") != "none"
                and (f.get("height") or 0) <= MAX_VIDEO_HEIGHT]
    playable = [f for f in combined if f.get("ext") == "mp4"] or combined
    # the generic extractor (tests) reports no codecs: one file with both
    unknown = [f for f in formats if not f.get("vcodec") and not f.get("acodec")]
    best_combined = max(playable or unknown, key=lambda f: (f.get("height") or 0, f.get("tbr") or 0), default=None)
    if best_audio is not None and (not video or best_video is not None):
        return Choice(best_audio, best_video if video else None, None)
    if best_combined is not None:
        return Choice(None, None, best_combined)
    if best_audio is not None:
        return Choice(best_audio, None, None)
    raise YouTubeError("YouTube offers no audio for this video that can be retrieved without DRM.",
                       code="youtube_drm")


def _caption_tracks(info: dict[str, Any]) -> list[dict[str, Any]]:
    """Creator captions, and automatic captions in the video's own language
    (never machine translations)."""
    tracks = []
    for language, formats in (info.get("subtitles") or {}).items():
        if language == "live_chat" or not formats:
            continue
        tracks.append({"language": language, "name": formats[0].get("name") or language, "kind": "manual",
                       "formats": sorted({f.get("ext") for f in formats if f.get("ext")})})
    spoken = info.get("language")
    for language, formats in (info.get("automatic_captions") or {}).items():
        original = language.endswith("-orig") or (spoken and language == spoken)
        if original and formats:
            tracks.append({"language": language.removesuffix("-orig"), "name": formats[0].get("name") or language,
                           "kind": "auto", "key": language,
                           "formats": sorted({f.get("ext") for f in formats if f.get("ext")})})
    seen, unique = set(), []
    for track in tracks:
        key = (track["language"], track["kind"])
        if key not in seen:
            seen.add(key)
            unique.append(track)
    return unique


def refusal(info: dict[str, Any]) -> dict[str, str] | None:
    live = info.get("live_status")
    if live in _LIVE:
        return {"code": "youtube_live", "message": _LIVE[live]}
    if info.get("availability") in _NEEDS_ACCOUNT:
        return {"code": "youtube_restricted", "message": "This video needs a YouTube account (private, members-only "
                "or paid). Verbatim does not sign in or bypass restrictions."}
    formats = info.get("formats") or []
    if formats and all(f.get("has_drm") for f in formats):
        return {"code": "youtube_drm", "message": "This video is protected by DRM and cannot be imported."}
    return None


def sponsor_chapters(info: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"start": c.get("start_time"), "end": c.get("end_time"), "title": c.get("title")}
            for c in info.get("chapters") or [] if SPONSOR_CHAPTER.search(c.get("title") or "")]


def inspect(target: MediaURL) -> dict[str, Any]:
    """What the user needs to decide: what the video is, its licence, whether
    it can be retrieved (and if not, why), its captions and the size."""
    info = _extract(target, {"skip_download": True})
    refused = refusal(info)
    sizes = {}
    if refused is None:
        for kind, want_video in (("video", True), ("audio", False)):
            try:
                sizes[kind] = choose_formats(info, video=want_video).approx_bytes
            except YouTubeError as exc:
                refused = {"code": exc.code, "message": exc.message}
    return {
        "url": target.url,
        "video_id": target.video_id or info.get("id"),
        "title": info.get("title"),
        "channel": info.get("channel") or info.get("uploader"),
        "channel_url": info.get("channel_url") or info.get("uploader_url"),
        "duration": info.get("duration"),
        "upload_date": info.get("upload_date"),
        "license": info.get("license"),
        "language": info.get("language"),
        "live_status": info.get("live_status"),
        "availability": info.get("availability"),
        "thumbnail": info.get("thumbnail"),
        "chapters": [{"start": c.get("start_time"), "end": c.get("end_time"), "title": c.get("title")}
                     for c in info.get("chapters") or []],
        "sponsor_chapters": sponsor_chapters(info),
        "captions": _caption_tracks(info),
        "approx_bytes": sizes,
        "refusal": refused,
        "ads_note": ADS_NOTE,
    }


# ----------------------------------------------------------------------------- media


#: progress(bytes done, exact total bytes or None)
ProgressFn = Callable[[int, int | None], None]


def _download_format(target: MediaURL, fmt: dict[str, Any], folder: Path, stem: str, progress: ProgressFn | None,
                     should_cancel: Callable[[], bool] | None, done: int, overall: int | None,
                     single: bool) -> Path:
    """``done``: bytes of earlier parts; ``overall``: exact total of all parts
    if YouTube reported every size exactly. A single part may report its exact
    size while downloading; an estimate is never shown as a total."""
    def hook(status: dict[str, Any]) -> None:
        if should_cancel is not None and should_cancel():
            raise Cancelled()
        if progress is not None and status.get("status") == "downloading":
            total = overall
            if total is None and single and status.get("total_bytes"):
                total = int(status["total_bytes"])
            progress(done + int(status.get("downloaded_bytes") or 0), total)

    _extract(target, {
        "format": fmt["format_id"], "paths": {"home": str(folder), "temp": str(folder)},
        "outtmpl": {"default": f"{stem}.%(ext)s"}, "progress_hooks": [hook], "skip_download": False,
    }, download=True)
    found = [p for p in folder.glob(f"{stem}.*") if not p.name.endswith((".part", ".ytdl"))]
    if not found:
        raise YouTubeError("The download finished without a media file.", code="youtube_error")
    return found[0]


def remux(video: Path, audio: Path, out: Path) -> Path:
    """One MP4 with the video and audio streams as they are (no re-encoding),
    packets interleaved by time."""
    import av

    tmp = out.with_suffix(".tmp.mp4")
    with av.open(str(video)) as vin, av.open(str(audio)) as ain, av.open(str(tmp), "w", format="mp4") as mux:
        vs, as_ = vin.streams.video[0], ain.streams.audio[0]
        ov, oa = mux.add_stream_from_template(vs), mux.add_stream_from_template(as_)

        def packets(container, stream, target, order):
            for packet in container.demux(stream):
                if packet.dts is None:
                    continue
                time = float(packet.dts * packet.time_base)
                yield time, order, packet, target

        for _time, _order, packet, target in heapq.merge(packets(vin, vs, ov, 0), packets(ain, as_, oa, 1),
                                                         key=lambda item: (item[0], item[1])):
            packet.stream = target
            mux.mux(packet)
    os.replace(tmp, out)
    return out


@dataclass
class Retrieved:
    path: Path
    has_video: bool
    bytes: int
    formats: list[str]


def retrieve(target: MediaURL, info: dict[str, Any], folder: Path, *, video: bool,
             progress: ProgressFn | None = None, should_cancel: Callable[[], bool] | None = None) -> Retrieved:
    """Download the chosen formats into ``folder`` (partial downloads are
    continued) and return one playable file. Re-running after an interruption
    reuses whatever was completed."""
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / ("media.mp4" if video else "media.m4a")
    choice = choose_formats(info, video=video)
    parts = [f for f in (choice.combined, choice.audio, choice.video if video else None) if f]
    sizes = [f.get("filesize") for f in parts]
    overall = int(sum(sizes)) if all(sizes) else None
    if choice.combined is not None:
        existing = [p for p in folder.glob("media.*") if p.suffix not in (".part", ".ytdl")]
        path = existing[0] if existing else _download_format(target, choice.combined, folder, "media", progress,
                                                             should_cancel, 0, overall, True)
        return Retrieved(path, True, path.stat().st_size, [str(choice.combined["format_id"])])
    if final.exists():  # completed by an earlier attempt
        return Retrieved(final, video, final.stat().st_size,
                         [str(f["format_id"]) for f in (choice.audio, choice.video) if f])
    audio = _download_format(target, choice.audio, folder, "audio", progress, should_cancel, 0, overall, not video)
    if not video:
        if audio.suffix == ".m4a":  # otherwise (e.g. Opus in WebM) it stays as served
            os.replace(audio, final)
            audio = final
        return Retrieved(audio, False, audio.stat().st_size, [str(choice.audio["format_id"])])
    video_file = _download_format(target, choice.video, folder, "video", progress, should_cancel,
                                  audio.stat().st_size, overall, False)
    remux(video_file, audio, final)
    audio.unlink(missing_ok=True)
    video_file.unlink(missing_ok=True)
    return Retrieved(final, True, final.stat().st_size, [str(choice.video["format_id"]), str(choice.audio["format_id"])])


def retrieve_captions(target: MediaURL, tracks: list[dict[str, Any]], folder: Path) -> list[dict[str, Any]]:
    """Download the chosen caption tracks as WebVTT and parse them."""
    if not tracks:
        return []
    folder.mkdir(parents=True, exist_ok=True)
    manual = [t["language"] for t in tracks if t["kind"] == "manual"]
    auto = [t.get("key", t["language"]) for t in tracks if t["kind"] == "auto"]
    _extract(target, {
        "skip_download": True, "writesubtitles": bool(manual), "writeautomaticsub": bool(auto),
        "subtitleslangs": [*manual, *auto], "subtitlesformat": "vtt/best",
        "paths": {"home": str(folder), "temp": str(folder)}, "outtmpl": {"default": "captions", "subtitle": "captions"},
    }, download=True)
    out = []
    for track in tracks:
        key = track.get("key", track["language"])
        files = sorted(folder.glob(f"captions.{key}.*"))
        vtt = [p for p in files if p.suffix == ".vtt"]
        if not vtt:
            log.info("no WebVTT captions for %s", key)
            continue
        cues: list[Cue] = parse_vtt(vtt[0].read_text(encoding="utf-8", errors="replace"))
        out.append({**{k: track[k] for k in ("language", "name", "kind")}, "cues": [c.to_dict() for c in cues]})
    return out
