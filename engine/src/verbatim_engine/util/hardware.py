"""Host hardware queries used for scheduling decisions."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from functools import lru_cache


def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(
            ["/usr/sbin/sysctl", "-n", name],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = out.stdout.strip()
    return value or None


@lru_cache(maxsize=1)
def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


@lru_cache(maxsize=1)
def macos_version() -> tuple[int, ...] | None:
    if sys.platform != "darwin":
        return None
    release = platform.mac_ver()[0]
    if not release:
        return None
    return tuple(int(p) for p in release.split(".") if p.isdigit())


@lru_cache(maxsize=1)
def performance_core_count() -> int:
    """Number of performance cores (Apple Silicon) or logical CPUs elsewhere.

    CTranslate2 runs fastest when its threads stay on performance cores; on an
    M2 Pro that is 6 or 8 of the 10/12 cores.
    """
    if sys.platform == "darwin":
        value = _sysctl("hw.perflevel0.physicalcpu")
        if value and value.isdigit() and int(value) > 0:
            return int(value)
    return max(1, os.cpu_count() or 1)


@lru_cache(maxsize=1)
def total_memory_bytes() -> int | None:
    if sys.platform == "darwin":
        value = _sysctl("hw.memsize")
        return int(value) if value and value.isdigit() else None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def describe() -> dict:
    return {
        "platform": sys.platform,
        "machine": platform.machine(),
        "macos_version": ".".join(map(str, macos_version())) if macos_version() else None,
        "apple_silicon": is_apple_silicon(),
        "performance_cores": performance_core_count(),
        "logical_cpus": os.cpu_count(),
        "memory_bytes": total_memory_bytes(),
        "python": platform.python_version(),
    }
