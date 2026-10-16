"""Capture real engine protocol messages as fixtures for the Swift client tests.

Starts ``verbatim-engine serve`` as a subprocess (exactly like the app),
transcribes the JFK corpus sample with a real model, and writes every message
type the app decodes to ``shared/protocol-fixtures/``. Paths that depend on the
machine are replaced with neutral placeholders; everything else is verbatim.

    uv run python scripts/capture_protocol_fixtures.py --model-path /path/to/faster-whisper-model \
        --diarization-dir /path/to/diarization/model --alignment-dir /path/to/alignment/model \
        --conversation /path/to/sample.wav

With the speaker and alignment models, a two-person conversation is also
transcribed with speakers and word alignment, and the correction methods
(rename, undo) are captured.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from queue import Queue

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "shared" / "protocol-fixtures"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--backend", default="faster-whisper")
    parser.add_argument("--diarization-dir", help="installed diarization model folder")
    parser.add_argument("--alignment-dir", help="installed English alignment model folder")
    parser.add_argument("--conversation", help="a short multi-speaker recording (e.g. the speaker corpus sample.wav)")
    args = parser.parse_args()

    data_dir = Path(tempfile.mkdtemp(prefix="verbatim-fixtures-"))

    def install_speaker_models() -> None:
        # Make the installed models visible to this fresh data folder. Done only
        # after the JFK part, whose fixtures show the app before the speaker and
        # alignment models are downloaded (both stages skipped, with a reason).
        for group, family, folder in (("diarization", "pyannote-wespeaker", args.diarization_dir),
                                      ("alignment", "en-conformer-small", args.alignment_dir)):
            target = data_dir / "models" / group / family
            target.mkdir(parents=True)
            for item in Path(folder).iterdir():
                (target / item.name).symlink_to(item.resolve())

    proc = subprocess.Popen(
        [sys.executable, "-m", "verbatim_engine", "serve", "--data-dir", str(data_dir)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
    )
    inbox: Queue = Queue()
    threading.Thread(target=lambda: [inbox.put(json.loads(l)) for l in proc.stdout], daemon=True).start()
    notifications: list[dict] = []
    counter = [0]

    def call(method: str, params: dict | None = None) -> dict:
        counter[0] += 1
        rid = counter[0]
        proc.stdin.write(json.dumps({"id": rid, "method": method, "params": params or {}}) + "\n")
        proc.stdin.flush()
        while True:
            msg = inbox.get(timeout=600)
            if msg.get("id") == rid:
                return msg
            if "id" not in msg:
                notifications.append(msg)

    OUT.mkdir(parents=True, exist_ok=True)

    replacements = [
        (args.model_path, "/Users/example/Models/faster-whisper-small"),
        *([(args.conversation, "/Users/example/Downloads/call.wav")] if args.conversation else []),
        (str(data_dir.resolve()), "/Users/example/Library/Application Support/Verbatim"),
        (str(data_dir), "/Users/example/Library/Application Support/Verbatim"),
        (str(ROOT), "/Users/example/verbatim"),
        (sys.executable, "/Applications/Verbatim.app/Contents/Resources/python/bin/python3"),
    ]

    def save(name: str, message: dict) -> None:
        # Only machine-specific paths are replaced; everything else is verbatim engine output.
        text = json.dumps(message, indent=2, sort_keys=True)
        for old, new in replacements:
            text = text.replace(json.dumps(old)[1:-1], json.dumps(new)[1:-1])
        (OUT / f"{name}.json").write_text(text + "\n")

    jfk = ROOT / "engine" / "tests" / "corpus" / "jfk.wav"
    save("engine.info", call("engine.info"))
    save("transcription.modes", call("transcription.modes"))
    save("models.list", call("models.list"))
    save("settings.get", call("settings.get"))
    save("error.model_not_installed", call("transcription.preflight", {"options": {"mode": "maximum", "backend": "faster-whisper"}}))
    save("error.not_found", call("projects.get", {"project_id": "missing"}))
    save("media.probe", call("media.probe", {"path": str(jfk)}))
    project = call("projects.create", {"path": str(jfk), "title": "JFK"})
    save("projects.create", project)
    options = {"backend": args.backend, "model_path": args.model_path, "mode": "balanced"}
    save("transcription.preflight", call("transcription.preflight", {"options": options}))
    job = call("jobs.start", {"project_id": project["result"]["id"], "options": options})
    save("jobs.start", job)
    job_id = job["result"]["id"]
    while True:
        snapshot = call("jobs.get", {"job_id": job_id})
        if snapshot["result"]["status"] in ("completed", "failed", "cancelled", "interrupted"):
            break
        time.sleep(0.3)
    save("jobs.get.completed", snapshot)
    running = next((n for n in notifications if n["method"] == "job.updated" and n["params"]["status"] == "running"
                    and any(s["status"] == "running" and s["fraction"] is not None for s in n["params"]["stages"])), None)
    if running:
        save("notification.job.updated.running", running)
    project_note = next((n for n in notifications if n["method"] == "project.updated"), None)
    if project_note:
        save("notification.project.updated", project_note)
    save("transcript.get", call("transcript.get", {"project_id": project["result"]["id"], "include_words": True}))
    save("export.render", call("export.render", {"project_id": project["result"]["id"], "format": "txt"}))
    if args.conversation and args.diarization_dir and args.alignment_dir:
        install_speaker_models()
        call_project = call("projects.create", {"path": args.conversation, "title": "Call"})["result"]
        speaker_options = {**options, "diarize": True, "align": True}
        job = call("jobs.start", {"project_id": call_project["id"], "options": speaker_options})["result"]
        while True:
            snapshot = call("jobs.get", {"job_id": job["id"]})
            if snapshot["result"]["status"] in ("completed", "failed", "cancelled", "interrupted"):
                break
            time.sleep(0.3)
        save("jobs.get.completed.speakers", snapshot)
        transcript = call("transcript.get", {"project_id": call_project["id"], "include_words": True})
        save("transcript.get.speakers", transcript)
        speaker = transcript["result"]["speakers"][0]
        save("transcript.renameSpeaker", call("transcript.renameSpeaker", {
            "project_id": call_project["id"], "speaker_id": speaker["id"], "name": "Diane"}))
        segments = transcript["result"]["segments"]
        splittable = next(s for s in segments if len(s["words"]) >= 2)
        save("transcript.splitSegment", call("transcript.splitSegment", {
            "project_id": call_project["id"], "segment_id": splittable["id"], "word_index": 1}))
        save("transcript.undo", call("transcript.undo", {"project_id": call_project["id"]}))
        save("transcript.history", call("transcript.history", {"project_id": call_project["id"]}))
        save("transcript.original", call("transcript.original", {"project_id": call_project["id"]}))
        save("error.speaker_not_found", call("transcript.renameSpeaker", {
            "project_id": call_project["id"], "speaker_id": 999999, "name": "x"}))
        note = next((n for n in notifications if n["method"] == "transcript.updated"), None)
        if note is None:
            time.sleep(0.5)
            note = next((n for n in notifications if n["method"] == "transcript.updated"), None)
        if note:
            save("notification.transcript.updated", note)
    save("projects.list", call("projects.list"))
    call("engine.shutdown")
    proc.wait(timeout=30)
    print(f"wrote fixtures to {OUT}")
    return 0 if snapshot["result"]["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
