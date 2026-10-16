"""Check every catalog entry against its source: HuggingFace repositories
exist and contain the needed files, fixed-URL files are reachable with the
recorded size, and the approximate download sizes shown to users are within
tolerance. Prints revisions and checksums so they can be pinned. Run in CI
(needs network access).

    uv run --project engine python scripts/verify_model_catalog.py
"""

from __future__ import annotations

import fnmatch
import sys

import httpx2
from huggingface_hub import HfApi

from verbatim_engine.asr.registry import backend_class
from verbatim_engine.models.catalog import ALL_MODELS

TOLERANCE = 0.20


def check_hf(api: HfApi, spec) -> bool:
    try:
        info = api.model_info(spec.repo_id, revision=spec.revision, files_metadata=True)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {spec.id}: {spec.repo_id} not reachable: {exc}")
        return False
    siblings = [s for s in info.siblings or [] if any(fnmatch.fnmatch(s.rfilename, p) for p in spec.allow_patterns)]
    files = {s.rfilename: s.size or 0 for s in siblings}
    total = sum(files.values())
    if spec.kind == "asr":
        required = backend_class(spec.backend).required_files
        missing = [" or ".join(group) for group in required if not any(name in files for name in group)]
    else:
        missing = [p for p in spec.required_files if not any(fnmatch.fnmatch(name, p) for name in files)]
    ratio = total / spec.approx_download_bytes if spec.approx_download_bytes else 0
    ok = not missing and abs(ratio - 1) <= TOLERANCE
    print(f"{spec.id:34} {spec.repo_id:48} {total / 1e9:8.3f}G {spec.approx_download_bytes / 1e9:8.3f}G  "
          f"{info.sha[:12]} {'ok' if ok else 'MISMATCH'}")
    for s in siblings:
        lfs = getattr(s, "lfs", None)
        if spec.kind != "asr" and lfs is not None:
            print(f"     {s.rfilename}: {s.size} bytes, sha256 {lfs.sha256}")
    if missing:
        print(f"     missing required files: {missing}")
    return ok


def check_urls(client: httpx2.Client, spec) -> bool:
    ok = True
    for remote in spec.files:
        try:
            response = client.head(remote.url, follow_redirects=True)
            size = int(response.headers.get("content-length", -1))
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {spec.id}: {remote.url} not reachable: {exc}")
            ok = False
            continue
        good = response.status_code == 200 and size == remote.size
        ok &= good
        print(f"{spec.id:34} {remote.url.rsplit('/', 1)[-1]:48} {size:>12} bytes {'ok' if good else 'MISMATCH'}")
    if sum(f.size for f in spec.files) != spec.approx_download_bytes:
        print(f"     catalog size {spec.approx_download_bytes} differs from the files' total")
        ok = False
    return ok


def main() -> int:
    api = HfApi()
    failures = 0
    with httpx2.Client(timeout=60) as client:
        for spec in ALL_MODELS:
            ok = check_urls(client, spec) if spec.files else check_hf(api, spec)
            failures += 0 if ok else 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
