"""Media validation and probing (Media Import Manager, engine side).

Media is opened in-process with PyAV (FFmpeg's libraries); no shell commands
and no external ``ffmpeg`` binary are involved.
"""

from __future__ import annotations

import os
import stat
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..errors import MediaError, NoAudioStream

#: Containers that are images or otherwise never carry a timeline.
_IMAGE_FORMATS = {"image2", "png_pipe", "jpeg_pipe", "gif", "webp_pipe", "bmp_pipe", "tiff_pipe", "svg_pipe"}


@dataclass
class AudioStreamInfo:
    index: int
    codec: str
    sample_rate: int
    channels: int
    duration: float | None
    language: str | None
    is_default: bool


@dataclass
class VideoStreamInfo:
    index: int
    codec: str
    width: int
    height: int
    frame_rate: float | None


@dataclass
class MediaInfo:
    path: str
    size_bytes: int
    mtime_ns: int
    format_name: str
    format_long_name: str
    duration: float | None
    start_time: float
    bit_rate: int | None
    audio_streams: list[AudioStreamInfo] = field(default_factory=list)
    video_streams: list[VideoStreamInfo] = field(default_factory=list)
    selected_audio_stream: int | None = None

    @property
    def has_video(self) -> bool:
        return bool(self.video_streams)

    @property
    def has_audio(self) -> bool:
        return bool(self.audio_streams)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["has_video"] = self.has_video
        data["has_audio"] = self.has_audio
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MediaInfo":
        data = dict(data)
        data.pop("has_video", None)
        data.pop("has_audio", None)
        data["audio_streams"] = [AudioStreamInfo(**s) for s in data.get("audio_streams", [])]
        data["video_streams"] = [VideoStreamInfo(**s) for s in data.get("video_streams", [])]
        return cls(**data)


def validate_path(path: str | os.PathLike[str]) -> Path:
    """Resolve and validate a user-supplied media path.

    Rejects anything that is not an existing, readable, non-empty regular file.
    """
    if not isinstance(path, (str, os.PathLike)) or not str(path).strip():
        raise MediaError("No file path was provided.")
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise MediaError("Media paths must be absolute.", data={"path": str(path)})
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError:
        raise MediaError("The file does not exist.", data={"path": str(path)}) from None
    except (OSError, RuntimeError) as exc:
        raise MediaError(f"The file path could not be resolved: {exc}", data={"path": str(path)}) from exc
    try:
        st = resolved.stat()
    except OSError as exc:
        raise MediaError(f"The file could not be read: {exc.strerror}", data={"path": str(path)}) from exc
    if not stat.S_ISREG(st.st_mode):
        raise MediaError("Only regular files can be imported.", data={"path": str(path)})
    if st.st_size == 0:
        raise MediaError("The file is empty.", data={"path": str(path)})
    if not os.access(resolved, os.R_OK):
        raise MediaError("The file is not readable (permission denied).", data={"path": str(path)})
    return resolved


def _seconds(value: int | None, time_base) -> float | None:
    if value is None or time_base is None:
        return None
    return float(value * time_base)


def probe(path: str | os.PathLike[str]) -> MediaInfo:
    """Validate and inspect a media file. Raises :class:`MediaError` if the file
    cannot be used for transcription."""
    import av  # imported lazily: keeps `serve` start-up fast

    resolved = validate_path(path)
    st = resolved.stat()
    try:
        container = av.open(str(resolved), mode="r", metadata_errors="ignore")
    except av.error.FFmpegError as exc:
        raise MediaError(
            "The file is not a recognised audio or video format.",
            data={"path": str(resolved), "detail": str(exc)},
        ) from exc
    try:
        fmt = container.format
        if fmt.name in _IMAGE_FORMATS:
            raise NoAudioStream("The file is an image, not audio or video.", data={"path": str(resolved)})

        duration = container.duration / 1_000_000 if container.duration else None
        start_time = container.start_time / 1_000_000 if container.start_time else 0.0

        audio: list[AudioStreamInfo] = []
        for stream in container.streams.audio:
            cc = stream.codec_context
            if not cc.sample_rate:
                continue
            channels = getattr(cc, "channels", None) or (cc.layout.nb_channels if cc.layout else 0)
            audio.append(
                AudioStreamInfo(
                    index=stream.index,
                    codec=cc.name,
                    sample_rate=int(cc.sample_rate),
                    channels=int(channels or 0),
                    duration=_seconds(stream.duration, stream.time_base),
                    language=(stream.metadata or {}).get("language"),
                    is_default=bool(stream.disposition & av.stream.Disposition.default),
                )
            )

        video: list[VideoStreamInfo] = []
        for stream in container.streams.video:
            # Cover art in audio files is exposed as a one-frame video stream.
            if stream.disposition & (av.stream.Disposition.attached_pic | av.stream.Disposition.still_image):
                continue
            cc = stream.codec_context
            rate = stream.average_rate
            video.append(
                VideoStreamInfo(
                    index=stream.index,
                    codec=cc.name,
                    width=int(cc.width or 0),
                    height=int(cc.height or 0),
                    frame_rate=float(rate) if rate else None,
                )
            )

        if not audio:
            raise NoAudioStream("The file contains no audio stream to transcribe.", data={"path": str(resolved)})

        if duration is None:
            durations = [s.duration for s in audio if s.duration]
            duration = max(durations) if durations else None

        selected = next((s.index for s in audio if s.is_default), audio[0].index)
        return MediaInfo(
            path=str(resolved),
            size_bytes=st.st_size,
            mtime_ns=st.st_mtime_ns,
            format_name=fmt.name,
            format_long_name=fmt.long_name or fmt.name,
            duration=duration,
            start_time=start_time,
            bit_rate=container.bit_rate or None,
            audio_streams=audio,
            video_streams=video,
            selected_audio_stream=selected,
        )
    finally:
        container.close()


def check_unchanged(path: str, size_bytes: int, mtime_ns: int) -> bool:
    """True if the file still exists with the size and mtime recorded at import."""
    try:
        st = Path(path).stat()
    except OSError:
        return False
    return st.st_size == size_bytes and st.st_mtime_ns == mtime_ns
