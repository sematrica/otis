"""Check every catalog entry against HuggingFace: the repository exists, it
contains the files the backend needs, and the approximate download size shown
to users is within tolerance. Run in CI (needs network access).

    uv run --project engine python scripts/verify_model_catalog.py
"""

from __future__ import annotations

import fnmatch
import sys

from huggingface_hub import HfApi

from verbatim_engine.asr.registry import backend_class
from verbatim_engine.models.catalog import CATALOG

TOLERANCE = 0.20


def main() -> int:
    api = HfApi()
    failures = 0
    print(f"{'model':32} {'repo':48} {'actual':>9} {'catalog':>9}  revision")
    for spec in CATALOG:
        try:
            info = api.model_info(spec.repo_id, files_metadata=True)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {spec.id}: {spec.repo_id} not reachable: {exc}")
            failures += 1
            continue
        files = {s.rfilename: s.size or 0 for s in info.siblings or []
                 if any(fnmatch.fnmatch(s.rfilename, p) for p in spec.allow_patterns)}
        total = sum(files.values())
        required = backend_class(spec.backend).required_files
        missing = [" or ".join(group) for group in required if not any(name in files for name in group)]
        ratio = total / spec.approx_download_bytes if spec.approx_download_bytes else 0
        ok = not missing and abs(ratio - 1) <= TOLERANCE
        print(f"{spec.id:32} {spec.repo_id:48} {total / 1e9:8.2f}G {spec.approx_download_bytes / 1e9:8.2f}G  "
              f"{info.sha[:12]} {'ok' if ok else 'MISMATCH'}")
        if missing:
            print(f"     missing required files: {missing}")
        if not ok:
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
