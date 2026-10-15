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
from .catalog import CATALOG, ModelSpec, get_spec

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
        return self.models_dir() / spec.backend / spec.family

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
        missing = backend_class(spec.backend).missing_files(path) if path.is_dir() else ["model folder"]
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
        return [{**spec.to_dict(), "status": self.status(spec)} for spec in CATALOG]

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
        return self.models_dir() / ".downloads" / f"{spec.backend}--{spec.family}"

    def download(
        self,
        model_id: str,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Download a catalog model (blocking). Intended to run in its own
        process so it can be cancelled by terminating that process; partial
        files are kept and resumed on the next attempt."""
        spec = get_spec(model_id)
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

        for name, size in expected.items():
            local = staging / name
            if not local.is_file():
                raise DownloadError(f"Downloaded model is missing {name}.")
            if size is not None and local.stat().st_size != size:
                raise DownloadError(f"Downloaded file {name} has the wrong size; please retry.")

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
