"""YouTube URL validation.

Only links to a single YouTube video are accepted, in their usual forms
(watch, youtu.be, shorts, live, embed, music and mobile sites), and they are
turned into one canonical ``https://www.youtube.com/watch?v=ID``. Anything
else — other sites, playlists without a video, channels, credentials or ports
in the URL, non-web schemes — is refused before any network access.

For tests only, ``VERBATIM_TEST_MEDIA_HOST`` may name one *loopback* host
(127.0.0.1, ::1 or localhost, with a port) whose pages are then accepted as
media pages; it can never open the import to another site.
"""

from __future__ import annotations

import ipaddress
import os
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit

from ..errors import InvalidParams

MAX_URL_LENGTH = 2048
VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_WATCH_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}
_EMBED_HOSTS = {"www.youtube-nocookie.com", "youtube-nocookie.com"}
_PATH_KINDS = ("shorts", "live", "embed", "v", "e")
TEST_HOST_VARIABLE = "VERBATIM_TEST_MEDIA_HOST"


@dataclass(frozen=True)
class MediaURL:
    #: the URL to retrieve (canonical for YouTube)
    url: str
    #: the YouTube video id, or None for the loopback test host
    video_id: str | None

    @property
    def is_youtube(self) -> bool:
        return self.video_id is not None


def _refuse(message: str) -> InvalidParams:
    return InvalidParams(message, data={"reason": "url"})


def _test_host() -> str | None:
    """The loopback host:port allowed for tests, if any."""
    value = os.environ.get(TEST_HOST_VARIABLE, "").strip().lower()
    if not value:
        return None
    host = urlsplit("http://" + value).hostname or ""
    if host == "localhost":
        return value
    try:
        return value if ipaddress.ip_address(host).is_loopback else None
    except ValueError:
        return None


def validate(text: str) -> MediaURL:
    """The canonical URL of a single YouTube video, or :class:`InvalidParams`."""
    raw = (text or "").strip()
    if not raw:
        raise _refuse("Paste the link of a YouTube video.")
    if len(raw) > MAX_URL_LENGTH or any(ch.isspace() for ch in raw):
        raise _refuse("This does not look like a YouTube link.")
    if "://" not in raw:
        raw = "https://" + raw
    parts = urlsplit(raw)
    if parts.scheme not in ("http", "https"):
        raise _refuse("Only web links (https://) to YouTube videos can be imported.")
    if parts.username or parts.password:
        raise _refuse("Links with a user name or password are not accepted.")
    host = (parts.hostname or "").lower().rstrip(".")
    test = _test_host()
    if test is not None and parts.netloc.lower() == test:
        return MediaURL(raw, None)
    if parts.port not in (None, 80, 443):
        raise _refuse("This does not look like a YouTube link.")
    video_id = None
    path = [p for p in parts.path.split("/") if p]
    query = parse_qs(parts.query)
    if host == "youtu.be" and path:
        video_id = path[0]
    elif host in _WATCH_HOSTS:
        if path[:1] == ["watch"]:
            video_id = (query.get("v") or [None])[0]
        elif len(path) >= 2 and path[0] in _PATH_KINDS:
            video_id = path[1]
        elif path[:1] == ["playlist"] or (query.get("list") and not query.get("v")):
            raise _refuse("Playlists cannot be imported. Open one video of the playlist and paste its link.")
        elif path and path[0] in ("channel", "c", "user") or (path and path[0].startswith("@")):
            raise _refuse("This is a channel, not a video. Paste the link of one video.")
    elif host in _EMBED_HOSTS and len(path) >= 2 and path[0] == "embed":
        video_id = path[1]
    else:
        raise _refuse("Only YouTube links can be imported. For other media, download it where you are allowed "
                      "to and import the file.")
    if not video_id or not VIDEO_ID.match(video_id):
        raise _refuse("This YouTube link does not name a video.")
    return MediaURL(f"https://www.youtube.com/watch?v={video_id}", video_id)
