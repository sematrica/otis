"""Backend discovery."""

from __future__ import annotations

from typing import Any

from ..errors import BackendUnavailable, InvalidParams
from .base import Backend
from .faster_whisper_backend import FasterWhisperBackend
from .mlx_whisper_backend import MlxWhisperBackend

BACKENDS: dict[str, type[Backend]] = {
    FasterWhisperBackend.name: FasterWhisperBackend,
    MlxWhisperBackend.name: MlxWhisperBackend,
}

#: Preference order when the user lets the engine choose.
AUTO_ORDER = (MlxWhisperBackend.name, FasterWhisperBackend.name)


def backend_class(name: str) -> type[Backend]:
    try:
        return BACKENDS[name]
    except KeyError:
        raise InvalidParams(f"Unknown backend '{name}'.", data={"known": sorted(BACKENDS)}) from None


def create_backend(name: str) -> Backend:
    cls = backend_class(name)
    available, reason = cls.availability()
    if not available:
        raise BackendUnavailable(reason or f"{name} is unavailable", data={"backend": name})
    return cls()


def describe_backends() -> list[dict[str, Any]]:
    out = []
    for name, cls in BACKENDS.items():
        available, reason = cls.availability()
        out.append(
            {
                "name": name,
                "display_name": cls.display_name,
                "available": available,
                "unavailable_reason": reason,
                "capabilities": cls.capabilities().to_dict(),
            }
        )
    return out
