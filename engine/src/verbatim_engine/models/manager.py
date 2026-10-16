"""Model Manager: where models live, whether they are installed, and explicit
download/delete.

A model directory counts as installed only if it contains a manifest written
after a verified download (or registration of a user-supplied folder) and all
files the backend needs. Nothing is ever downloaded implicitly.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Callable

from ..asr.registry import backend_class
from ..errors import DownloadError, InvalidParams, ModelNotInstalled
from ..paths import EnginePaths
from ..storage import repo
from ..storage.db import Database
from ..util.ids import utc_now
from .catalog import ALL_MODELS, ModelSpec, RemoteFile, get_spec

log = logging.getLogger(__name__)

MANIFEST_NAME = "verbatim-model.json"
SETTING_MODELS_DIR = "models_dir"


def dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


class ModelManager:
    def __init__(self, db: Database, paths: EnginePaths):
        self.db = db
        self.paths = paths

    # ------------------------------------------------------------------ location

    def models_dir(self) -> Path:
        configured = repo.get_setting(self.db, SETTING_MODELS_DIR)
        return Path(configured) if configured else self.paths.default_models_dir

    def set_models_dir(self, path: str | None) -> Path:
        if path in (None, ""):
            repo.set_setting(self.db, SETTING_MODELS_DIR, None)
            return self.models_dir()
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            raise InvalidParams("The models folder must be an absolute path.")
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise InvalidParams(f"The models folder cannot be created: {exc.strerror}") from exc
        if not candidate.is_dir() or not os.access(candidate, os.W_OK):
            raise InvalidParams("The models folder is not writable.")
        repo.set_setting(self.db, SETTING_MODELS_DIR, str(candidate.resolve()))
        return self.models_dir()

    def install_path(self, spec: ModelSpec) -> Path:
        return self.models_dir() / spec.install_group / spec.family

    @staticmethod
    def missing_files(spec: ModelSpec, path: Path) -> list[str]:
        if spec.kind == "asr":
            return backend_class(spec.backend).missing_files(path)
        return [name for name in spec.required_files if not any(path.glob(name))]

    # ------------------------------------------------------------------ status

    def status(self, spec: ModelSpec) -> dict[str, Any]:
        path = self.install_path(spec)
        manifest_path = path / MANIFEST_NAME
        manifest = None
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, ValueError):
                manifest = None
        missing = self.missing_files(spec, path) if path.is_dir() else ["model folder"]
        installed = manifest is not None and not missing
        return {
            "installed": installed,
            "path": str(path),
            "size_on_disk": dir_size(path) if path.is_dir() else 0,
            "revision": manifest.get("revision") if manifest else None,
            "downloaded_at": manifest.get("downloaded_at") if manifest else None,
            "missing": [] if installed else missing,
        }

    def list(self) -> list[dict[str, Any]]:
        return [{**spec.to_dict(), "status": self.status(spec)} for spec in ALL_MODELS]

    def installed(self, kind: str) -> list[ModelSpec]:
        return [spec for spec in ALL_MODELS if spec.kind == kind and self.status(spec)["installed"]]

    def require_installed(self, spec: ModelSpec) -> Path:
        status = self.status(spec)
        if not status["installed"]:
            raise ModelNotInstalled(
                f"{spec.display_name} ({spec.backend}) is not installed.",
                data={"model_id": spec.id, "approx_download_bytes": spec.approx_download_bytes},
            )
        return Path(status["path"])

    def delete(self, spec: ModelSpec) -> dict[str, Any]:
        path = self.install_path(spec)
        root = self.models_dir().resolve()
        if path.exists():
            resolved = path.resolve()
            if root not in resolved.parents:
                raise InvalidParams("Refusing to delete a folder outside the models directory.")
            shutil.rmtree(resolved)
        staging = self._staging_path(spec)
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        return self.status(spec)

    # ------------------------------------------------------------------ download

    def _staging_path(self, spec: ModelSpec) -> Path:
        return self.models_dir() / ".downloads" / f"{spec.install_group}--{spec.family}"

    def download(
        self,
        model_id: str,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Download a catalog model (blocking). Intended to run in its own
        process so it can be cancelled by terminating that process; partial
        files are kept and resumed on the next attempt."""
        spec = get_spec(model_id)
        if spec.files:
            return self._download_files(spec, progress)
        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError as exc:  # pragma: no cover - dependency is mandatory
            raise DownloadError("huggingface_hub is not installed.") from exc

        try:
            info = HfApi().model_info(spec.repo_id, revision=spec.revision, files_metadata=True)
        except Exception as exc:  # noqa: BLE001 - network/HTTP errors vary by version
            raise DownloadError(f"Could not reach the model repository: {exc}", data={"repo_id": spec.repo_id}) from exc

        files = [
            s for s in (info.siblings or []) if any(fnmatch.fnmatch(s.rfilename, p) for p in spec.allow_patterns)
        ]
        if not files:
            raise DownloadError("The model repository contains none of the expected files.", data={"repo_id": spec.repo_id})
        expected = {s.rfilename: s.size for s in files}
        total = sum(size or 0 for size in expected.values())
        revision = info.sha

        staging = self._staging_path(spec)
        staging.mkdir(parents=True, exist_ok=True)

        stop = threading.Event()

        def poll() -> None:
            while not stop.wait(0.5):
                if progress is not None:
                    progress(min(dir_size(staging), total), total)

        poller = threading.Thread(target=poll, daemon=True)
        poller.start()
        try:
            snapshot_download(
                spec.repo_id,
                revision=revision,
                local_dir=str(staging),
                allow_patterns=list(spec.allow_patterns),
                max_workers=4,
            )
        except Exception as exc:  # noqa: BLE001
            raise DownloadError(f"Download failed: {exc}", data={"repo_id": spec.repo_id}) from exc
        finally:
            stop.set()
            poller.join(timeout=2)

        checksums = {s.rfilename: s.lfs.sha256 for s in files if getattr(s, "lfs", None) is not None}
        for name, size in expected.items():
            local = staging / name
            if not local.is_file():
                raise DownloadError(f"Downloaded model is missing {name}.")
            if size is not None and local.stat().st_size != size:
                raise DownloadError(f"Downloaded file {name} has the wrong size; please retry.")
            if name in checksums and _sha256(local) != checksums[name]:
                local.unlink()
                raise DownloadError(f"Downloaded file {name} is corrupt (checksum mismatch); please retry.")

        shutil.rmtree(staging / ".cache", ignore_errors=True)
        manifest = {
            "model_id": spec.id,
            "repo_id": spec.repo_id,
            "revision": revision,
            "files": [{"name": n, "size": s} for n, s in sorted(expected.items())],
            "downloaded_at": utc_now(),
        }
        (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))

        final = self.install_path(spec)
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            shutil.rmtree(final)
        os.replace(staging, final)
        if progress is not None:
            progress(total, total)
        status = self.status(spec)
        if not status["installed"]:
            raise DownloadError("The downloaded model failed validation.", data={"missing": status["missing"]})
        return status

    def _download_files(self, spec: ModelSpec, progress: Callable[[int, int], None] | None) -> dict[str, Any]:
        """Download fixed-URL files with resume, verify their SHA-256 and
        extract the needed archive members."""
        staging = self._staging_path(spec)
        partial = staging / ".partial"
        partial.mkdir(parents=True, exist_ok=True)
        total = sum(f.size for f in spec.files)
        done_before = 0
        for remote in spec.files:
            target = partial / Path(remote.url).name
            _fetch(remote, target, lambda n, base=done_before: progress(base + n, total) if progress else None)
            done_before += remote.size
            if remote.members:
                _extract_members(target, remote.members, staging)
            else:
                os.replace(target, staging / (remote.name or Path(remote.url).name))
        shutil.rmtree(partial, ignore_errors=True)
        manifest = {
            "model_id": spec.id,
            "revision": None,
            "files": [{"url": f.url, "size": f.size, "sha256": f.sha256} for f in spec.files],
            "downloaded_at": utc_now(),
        }
        (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
        final = self.install_path(spec)
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            shutil.rmtree(final)
        os.replace(staging, final)
        if progress is not None:
            progress(total, total)
        status = self.status(spec)
        if not status["installed"]:
            raise DownloadError("The downloaded model failed validation.", data={"missing": status["missing"]})
        return status

    # ------------------------------------------------------------------ custom models

    @staticmethod
    def validate_custom_dir(backend: str, path: str) -> Path:
        """Validate a user-supplied model folder (e.g. a fine-tuned model)."""
        directory = Path(path).expanduser()
        if not directory.is_absolute() or not directory.is_dir():
            raise ModelNotInstalled("The model folder does not exist.", data={"path": path})
        missing = backend_class(backend).missing_files(directory)
        if missing:
            raise ModelNotInstalled(
                f"The model folder is missing {', '.join(missing)}.", data={"path": path, "missing": missing}
            )
        return directory.resolve()


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(remote: RemoteFile, target: Path, progress: Callable[[int], None]) -> None:
    """Download ``remote`` to ``target``, resuming a partial file, and verify it."""
    import httpx2

    if target.is_file() and target.stat().st_size == remote.size and _sha256(target) == remote.sha256:
        progress(remote.size)
        return
    have = target.stat().st_size if target.is_file() else 0
    if have >= remote.size:
        target.unlink()
        have = 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    try:
        with httpx2.Client(follow_redirects=True, timeout=httpx2.Timeout(30.0, read=120.0)) as client:
            with client.stream("GET", remote.url, headers=headers) as response:
                if response.status_code == 200:
                    have = 0  # server ignored the range: start over
                elif response.status_code != 206:
                    raise DownloadError(f"Download failed (HTTP {response.status_code}).", data={"url": remote.url})
                with open(target, "ab" if have else "wb") as out:
                    for block in response.iter_bytes(1 << 20):
                        out.write(block)
                        have += len(block)
                        progress(min(have, remote.size))
                        if have > remote.size:
                            raise DownloadError("The server sent more data than expected.", data={"url": remote.url})
    except DownloadError:
        raise
    except Exception as exc:  # noqa: BLE001 - network errors vary
        raise DownloadError(f"Download failed: {exc}", data={"url": remote.url}) from exc
    if target.stat().st_size != remote.size:
        raise DownloadError("The download was incomplete; please retry.", data={"url": remote.url})
    if _sha256(target) != remote.sha256:
        target.unlink()
        raise DownloadError("The downloaded file is corrupt (checksum mismatch); please retry.", data={"url": remote.url})


def _extract_members(archive: Path, members: tuple[tuple[str, str], ...], destination: Path) -> None:
    """Extract only the named regular files of a verified archive."""
    import tarfile

    wanted = dict(members)
    found: set[str] = set()
    with tarfile.open(archive, "r:bz2") as tar:
        for info in tar:
            if info.name not in wanted:
                continue
            if not info.isfile():
                raise DownloadError(f"Unexpected archive entry {info.name}.")
            source = tar.extractfile(info)
            if source is None:
                raise DownloadError(f"Unreadable archive entry {info.name}.")
            out_path = destination / wanted[info.name]
            with source, open(out_path, "wb") as out:
                shutil.copyfileobj(source, out, 1 << 20)
            found.add(info.name)
    missing = set(wanted) - found
    if missing:
        raise DownloadError(f"The archive is missing {', '.join(sorted(missing))}.")
