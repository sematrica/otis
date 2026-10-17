"""Command-line entry points.

    verbatim-engine serve                 JSON-lines RPC server used by the app
    verbatim-engine transcribe FILE       headless transcription to TXT
    verbatim-engine info                  engine, hardware and backend report
    verbatim-engine models                list catalog models and install state
    verbatim-engine download-model ID     explicit model download
    verbatim-engine worker --job ID       (internal) run one job
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from typing import Any

from . import __version__


def _add_data_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default=None, help="engine data directory (default: platform location)")


def _cmd_transcribe(args: argparse.Namespace) -> int:
    from .errors import EngineError
    from .export import write_atomic
    from .jobs.options import normalize_options, resolve
    from .rpc.service import EngineService
    from .paths import EnginePaths
    from .storage import repo

    finished = threading.Event()
    last_line = [""]
    job_holder: dict[str, Any] = {}

    def notify(method: str, params: dict[str, Any]) -> None:
        if method != "job.updated" or params.get("id") != job_holder.get("id"):
            return
        stage = next((s for s in params["stages"] if s["status"] == "running"), None)
        if stage is not None:
            pct = f" {stage['fraction'] * 100:5.1f}%" if stage["fraction"] is not None else ""
            eta = f" eta {stage['eta_seconds']:.0f}s" if stage.get("eta_seconds") is not None else ""
            line = f"{stage['label']}{pct}{eta}"
            if line != last_line[0]:
                print(line, file=sys.stderr, flush=True)
                last_line[0] = line
        if params["status"] in ("completed", "failed", "cancelled", "interrupted"):
            finished.set()

    import os

    def absolute(path: str | None) -> str | None:
        return os.path.abspath(os.path.expanduser(path)) if path else None

    args.file, args.model_path, args.output = absolute(args.file), absolute(args.model_path), absolute(args.output)
    paths = EnginePaths.create(args.data_dir)
    service = EngineService(paths, notify)
    service.start()
    try:
        options = normalize_options(
            {
                "mode": args.mode,
                "backend": args.backend,
                "model_id": args.model_id,
                "model_path": args.model_path,
                "language": args.language,
                "chunk_seconds": args.chunk_seconds,
                "diarize": {"auto": "auto", "on": True, "off": False}[args.speakers],
                "speakers": {"num": args.num_speakers, "min": args.min_speakers, "max": args.max_speakers},
                **({"align": {"auto": "auto", "on": True, "off": False}[args.align]} if args.align else {}),
                "alignment_model": args.alignment_model,
            }
        )
        resolved = resolve(options, service.models)
        for warning in resolved.warnings:
            print(f"note: {warning}", file=sys.stderr)
        project = service.projects_create({"path": args.file})
        job = service.jobs.submit(project["id"], options)
        job_holder["id"] = job["id"]
        while not finished.wait(0.5):
            current = repo.get_job(service.db, job["id"])
            if current["status"] in ("completed", "failed", "cancelled", "interrupted"):
                break
        job = repo.get_job(service.db, job["id"])
        if job["status"] != "completed":
            print(f"error: {job['error']['message'] if job['error'] else job['status']}", file=sys.stderr)
            return 1
        text = service.export_render({"project_id": project["id"], "format": args.format})["content"]
        if args.output:
            write_atomic(args.output, text)
            print(f"wrote {args.output}", file=sys.stderr)
        else:
            sys.stdout.write(text)
        if args.json:
            print(json.dumps({"project_id": project["id"], "job": job}, indent=2), file=sys.stderr)
        return 0
    except EngineError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1
    finally:
        service.shutdown()


def _cmd_info(args: argparse.Namespace) -> int:
    from .rpc.service import EngineService
    from .paths import EnginePaths

    service = EngineService(EnginePaths.create(args.data_dir), lambda *_: None)
    try:
        print(json.dumps(service.engine_info({}), indent=2))
    finally:
        service.db.close()
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    from .rpc.service import EngineService
    from .paths import EnginePaths

    service = EngineService(EnginePaths.create(args.data_dir), lambda *_: None)
    try:
        listing = service.models_list({})
        print(f"models folder: {listing['models_dir']}")
        for model in listing["models"]:
            state = "installed" if model["status"]["installed"] else "not installed"
            size = model["approx_download_bytes"]
            shown = f"{size / 1e9:4.1f} GB" if size >= 1e9 else f"{size / 1e6:4.0f} MB"
            print(f"  {model['id']:<34} ~{shown:>7}  {state}")
    finally:
        service.db.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="verbatim-engine", description="Verbatim transcription engine")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="run the JSON-lines RPC server on stdin/stdout")
    _add_data_dir(p)

    p = sub.add_parser("worker", help=argparse.SUPPRESS)
    p.add_argument("--job", required=True)
    _add_data_dir(p)

    p = sub.add_parser("download-model", help="download a catalog model (explicit, user-initiated)")
    p.add_argument("model_id")
    _add_data_dir(p)

    p = sub.add_parser("transcribe", help="transcribe a file and print or write the transcript")
    p.add_argument("file")
    p.add_argument("--mode", default="balanced", choices=["fast", "balanced", "maximum"])
    p.add_argument("--backend", default="auto", choices=["auto", "faster-whisper", "mlx"])
    p.add_argument("--model-id", default=None)
    p.add_argument("--model-path", default=None, help="custom model folder (requires --backend)")
    p.add_argument("--language", default=None, help="ISO code; default: detect")
    p.add_argument("--chunk-seconds", type=float, default=600.0)
    p.add_argument("--speakers", default="auto", choices=["auto", "on", "off"],
                   help="speaker identification (auto: when the model is installed)")
    p.add_argument("--num-speakers", type=int, default=None, help="exact number of speakers, if known")
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--max-speakers", type=int, default=None)
    p.add_argument("--align", default=None, choices=["auto", "on", "off"],
                   help="forced word alignment (default: per mode)")
    p.add_argument("--alignment-model", default=None, help="catalog id of the alignment model to use")
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--format", "-f", default="txt", choices=["txt", "srt", "vtt", "json", "csv", "md"])
    p.add_argument("--json", action="store_true", help="print job details to stderr")
    _add_data_dir(p)

    for name in ("info", "models"):
        p = sub.add_parser(name)
        _add_data_dir(p)

    args = parser.parse_args(argv)

    from .paths import EnginePaths
    from .util import logs

    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    if args.command in ("serve", "worker", "download-model"):
        logs.configure(EnginePaths.create(args.data_dir).logs_dir, args.command, level)
    else:
        logs.configure(None, args.command, logging.WARNING if level == logging.INFO else level)

    if args.command == "serve":
        from .rpc.server import serve

        return serve(args.data_dir)
    if args.command == "worker":
        from .jobs.worker import main as worker_main

        return worker_main(args.job, args.data_dir)
    if args.command == "download-model":
        from .models.downloads import run_download

        return run_download(args.model_id, args.data_dir)
    if args.command == "transcribe":
        return _cmd_transcribe(args)
    if args.command == "info":
        return _cmd_info(args)
    if args.command == "models":
        return _cmd_models(args)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
