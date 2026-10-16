#!/usr/bin/env python3
"""Check how well the review flags point at wrong speaker labels.

Runs the full pipeline (Whisper → speaker identification → word alignment →
speaker assignment) on every recording of an annotated corpus (``*.wav`` with
a matching ``*.rttm`` reference, e.g. the corpus from
``fetch_speaker_corpus.py``) and scores every word against the reference:

* *right* / *wrong* - one reference speaker is talking at the word's midpoint
  and the word's speaker maps (by the DER mapping) to that speaker or not;
* *ambiguous* - the reference has overlapping speech or silence there.

It reports what share of right and wrong words sit in a segment flagged for
review (uncertain speaker, uncertain speaker change, overlap, no speaker), and
the words next to speaker changes with their evidence.

    python scripts/evaluate_speaker_review.py --whisper /path/to/faster-whisper-small \
        --diarization /path/to/diarization/pyannote-wespeaker \
        --alignment /path/to/alignment/en-conformer-small --corpus /path/to/speaker-corpus [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import Counter
from pathlib import Path

from verbatim_engine.diarization.assign import word_evidence
from verbatim_engine.diarization.pipeline import DiarizationResult
from verbatim_engine.metrics.der import diarization_error_rate, parse_rttm
from verbatim_engine.models.catalog import get_spec
from verbatim_engine.models.manager import MANIFEST_NAME
from verbatim_engine.paths import EnginePaths
from verbatim_engine.rpc.service import EngineService
from verbatim_engine.storage import repo

REVIEW_FLAGS = {"speaker_uncertain", "speaker_change_uncertain", "overlapping_speech", "speaker_unassigned"}


def link_model(service: EngineService, spec_id: str, folder: Path) -> None:
    """Make an installed model folder visible to a fresh data folder."""
    spec = get_spec(spec_id)
    target = service.models.install_path(spec)
    target.mkdir(parents=True, exist_ok=True)
    for item in folder.iterdir():
        if item.name != MANIFEST_NAME:
            (target / item.name).symlink_to(item.resolve())
    (target / MANIFEST_NAME).write_text(json.dumps({"model_id": spec.id, "source": str(folder)}))


def evaluate(name: str, wav: Path, turns: list, args: argparse.Namespace, data_dir: Path) -> list[dict]:
    paths = EnginePaths.create(data_dir)
    service = EngineService(paths, lambda method, params: None)
    link_model(service, "diarization/pyannote-wespeaker", args.diarization)
    link_model(service, "alignment/en-conformer-small", args.alignment)
    service.start()
    try:
        project = service.projects_create({"path": str(wav), "title": name})
        options = {"backend": "faster-whisper", "model_path": str(args.whisper), "mode": "balanced",
                   "language": "en", "diarize": True, "align": True}
        job = service.jobs_start({"project_id": project["id"], "options": options})
        while (job := repo.get_job(service.db, job["id"]))["status"] in ("queued", "running"):
            time.sleep(0.2)
        if job["status"] != "completed":
            raise RuntimeError(f"{name}: job {job['status']}: {job['error']}")
        transcript = repo.get_transcript(service.db, project["id"], include_words=True)
        cluster = {row["id"]: row["cluster_index"]
                   for row in service.db.query("SELECT id, cluster_index FROM speakers WHERE project_id = ?",
                                               (project["id"],))}
    finally:
        service.shutdown()
    npz = paths.job_dir(job["id"]) / "diarization.npz"
    if not transcript["segments"] or not npz.exists():
        return []
    diar = DiarizationResult.load(npz)
    labels = {sp["id"]: sp["label"] for sp in transcript["speakers"]}
    hypothesis = [(s["start"], s["end"], labels[s["speaker_id"]]) for s in transcript["segments"]
                  if s["speaker_id"] is not None]
    mapping = diarization_error_rate(turns, hypothesis).mapping
    rows = []
    for seg in transcript["segments"]:
        flags = [f["type"] for f in seg["flags"]]
        split = next((f for f in seg["flags"] if f["type"] == "split_at_speaker_change"), None)
        words = seg["words"]
        for i, w in enumerate(words):
            mid = (w["start"] + w["end"]) / 2
            active = sorted({label for a, b, label in turns if a <= mid < b})
            hyp = mapping.get(labels.get(w["speaker_id"]))
            status = "ambiguous" if len(active) != 1 else ("right" if hyp == active[0] else "wrong")
            at_change = bool(split) and ((split["part"] > 1 and i == 0)
                                         or (split["part"] < split["parts"] and i == len(words) - 1))
            ev = word_evidence(diar, w["start"], w["end"])
            rows.append({"recording": name, "text": w["text"].strip(), "start": w["start"], "status": status,
                         "flagged": bool(REVIEW_FLAGS & set(flags)), "segment_flags": flags, "at_change": at_change,
                         "coverage": round(ev.coverage, 3), "overlap": round(ev.overlap, 3),
                         "agreement": round(ev.agreement, 3),
                         "margin": None if ev.margin is None else round(ev.margin, 3),
                         "evidence_agrees": ev.speaker == cluster.get(w["speaker_id"])})
    return rows


def summarise(rows: list[dict]) -> dict:
    out: dict = {}
    for status in ("right", "wrong", "ambiguous"):
        group = [r for r in rows if r["status"] == status]
        out[status] = {"words": len(group), "in_flagged_segment": sum(r["flagged"] for r in group)}
    changes = [r for r in rows if r["at_change"]]
    out["words_at_speaker_changes"] = {
        "words": len(changes), "flagged_uncertain": sum("speaker_change_uncertain" in r["segment_flags"] for r in changes),
        **Counter(r["status"] for r in changes)}
    out["wrong_by_recording"] = dict(Counter(r["recording"] for r in rows if r["status"] == "wrong"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--whisper", type=Path, required=True, help="faster-whisper model folder")
    parser.add_argument("--diarization", type=Path, required=True, help="installed diarization model folder")
    parser.add_argument("--alignment", type=Path, required=True, help="installed Conformer-CTC Small folder")
    parser.add_argument("--corpus", type=Path, required=True, help="folder with *.wav and *.rttm references")
    parser.add_argument("--json", type=Path, help="write the per-word rows and the summary here")
    args = parser.parse_args()

    references: dict[str, list] = {}
    for rttm in sorted(args.corpus.glob("*.rttm")):
        references.update(parse_rttm(rttm.read_text()))
    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="verbatim-review-") as tmp:
        for name in sorted(references):
            wav = args.corpus / f"{name}.wav"
            if wav.exists():
                found = evaluate(name, wav, references[name], args, Path(tmp) / name)
                print(f"{name}: {len(found)} words", flush=True)
                rows.extend(found)
    summary = summarise(rows)
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "words": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
