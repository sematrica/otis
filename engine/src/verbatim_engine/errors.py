"""Engine error types.

Every error that crosses the RPC boundary carries a stable machine-readable
``code`` so the app can react (e.g. offer a model download) without parsing
human-readable messages.
"""

from __future__ import annotations

from typing import Any


class EngineError(Exception):
    """An expected, user-reportable failure."""

    code = "internal_error"

    def __init__(self, message: str, *, code: str | None = None, data: dict[str, Any] | None = None):
        super().__init__(message)
        if code is not None:
            self.code = code
        self.message = message
        self.data = data or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data:
            payload["data"] = self.data
        return payload


class InvalidParams(EngineError):
    code = "invalid_params"


class NotFound(EngineError):
    code = "not_found"


class MediaError(EngineError):
    """The file is missing, unreadable, or not decodable media."""

    code = "media_invalid"


class NoAudioStream(MediaError):
    code = "media_no_audio"


class SourceChanged(EngineError):
    """The source file was modified or removed after the project was created."""

    code = "source_changed"


class ModelNotInstalled(EngineError):
    code = "model_not_installed"


class BackendUnavailable(EngineError):
    code = "backend_unavailable"


class InvalidJobState(EngineError):
    code = "invalid_job_state"


class Cancelled(EngineError):
    code = "cancelled"


class DownloadError(EngineError):
    code = "download_failed"
