#!/usr/bin/env python3
"""Accuracy benchmark: the real pipeline on the Stage 3 corpus.

Every item of ``tests/support/corpus.py`` is transcribed through the engine
service exactly as the app does it (job, worker process, all passes), once per
*profile*:

* ``single`` - one recognition pass, plus quality checks and validation;
* ``multi`` - the same with targeted re-transcription of doubtful passages,
  judged by acoustic evidence (passes 6-7, as in Maximum Accuracy mode).

Both profiles use the same Whisper model, so the difference between them is
the effect of the extra passes. Measured per item: WER and CER (verified
transcript), DER (speaker annotation, collar 0.25 s), word-start error (exact
word times), words produced on silence, and how the review flags relate to the
words that are actually wrong. ``--interrupted`` adds a run that is stopped
mid-transcription (the app quitting), resumed, and compared.

    python scripts/benchmark_accuracy.py --whisper /path/to/faster-whisper-model \
        --diarization /path/to/diarization/pyannote-wespeaker --alignment /path/to/alignment/en-conformer-small \
        [--speaker-corpus /path/to/speaker-corpus] [--profiles single,multi] [--long] [--interrupted] \
        [--items jfk,call] [--json out.json] [--markdown out.md]
"""

from __future__ import annotations

import argparse
import difflib
import json
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tests"))

from support import corpus as corpus_module  # noqa: E402

from verbatim_engine.metrics.der import diarization_error_rate  # noqa: E402
from verbatim_engine.metrics.wer import character_error_rate, normalize, word_error_rate  # noqa: E402
from verbatim_engine.models.catalog import get_spec  # noqa: E402
from verbatim_engine.models.manager import MANIFEST_NAME  # noqa: E402
from verbatim_engine.paths import EnginePaths  # noqa: E402
from verbatim_engine.rpc.service import EngineService  # noqa: E402
from verbatim_engine.storage import repo  # noqa: E402
from verbatim_engine.verify.review import LEVELS, review_priority  # noqa: E402

TERMINAL = ("completed", "failed", "cancelled", "interrupted")
PROFILES = {"single": {"recheck": False}, "multi": {"recheck": True}}


def link_model(service: EngineService, spec_id: str, folder: Path) -> None:
    spec = get_spec(spec_id)
    target = service.models.install_path(spec)
    target.mkdir(parents=True, exist_ok=True)
    for item in folder.iterdir():
        if item.name != MANIFEST_NAME and not (target / item.name).exists():
            (target / item.name).symlink_to(item.resolve())
    (target / MANIFEST_NAME).write_text(json.dumps({"model_id": spec.id, "source": str(folder)}))


def make_service(data_dir: Path, args) -> EngineService:
    service = EngineService(EnginePaths.create(data_dir), lambda *_: None)
    if args.diarization:
        link_model(service, "diarization/pyannote-wespeaker", args.diarization)
    if args.alignment:
        family = "mms-300m" if any(args.alignment.glob("*.onnx")) and not (args.alignment / "tokens.txt").exists() \
            else "en-conformer-small"
        link_model(service, f"alignment/{family}", args.alignment)
    service.start()
    return service


def wait(service: EngineService, job_id: str, predicate=None) -> dict:
    while True:
        job = repo.get_job(service.db, job_id)
        if job["status"] in TERMINAL or (predicate is not None and predicate(job)):
            return job
        time.sleep(0.25)


def job_options(args, item, profile: str, **extra) -> dict:
    options = {"backend": args.backend, "model_path": str(args.whisper), "mode": args.mode, **PROFILES[profile], **extra}
    if args.force_language and item.language:
        options["language"] = item.language
    return options


# ----------------------------------------------------------------------------- metrics


#: Words of one speaker closer than this form one stretch of speech.
SPAN_GAP = 0.3


def speaker_spans(transcript: dict) -> list[tuple[float, float, str]]:
    """Who speaks when, as the transcript labels it: each speaker's words
    (pauses longer than SPAN_GAP between them are not speech)."""
    names = {sp["id"]: sp["label"] for sp in transcript["speakers"]}
    spans: list[tuple[float, float, str]] = []
    for seg in transcript["segments"]:
        if seg["speaker_id"] is None:
            continue
        label = names[seg["speaker_id"]]
        times = [(w["start"], w["end"]) for w in seg["words"] if w.get("start") is not None] or [(seg["start"], seg["end"])]
        start, end = times[0]
        for a, b in times[1:]:
            if a - end > SPAN_GAP:
                spans.append((start, end, label))
                start = a
            end = max(end, b)
        spans.append((start, end, label))
    return spans


def timing_errors(words: list[dict], reference: list[tuple[float, float, str]]) -> dict | None:
    hyp = [(w["start"], normalize(w["text"])) for w in words]
    ref = [(a, normalize(t)) for a, _b, t in reference]
    matcher = difflib.SequenceMatcher(None, [h for _, h in hyp], [r for _, r in ref], autojunk=False)
    errors = [abs(hyp[b.a + k][0] - ref[b.b + k][0]) for b in matcher.get_matching_blocks() for k in range(b.size)]
    if not errors:
        return None
    errors.sort()
    return {"matched_words": len(errors), "start_mean_abs_ms": round(1000 * sum(errors) / len(errors), 1),
            "start_within_50ms": round(sum(e <= 0.05 for e in errors) / len(errors), 3)}


def word_correctness(segments: list[dict], reference: str) -> list[tuple[dict, dict, bool]]:
    """``[(word, segment, correct)]``: a recognised word is correct if it is
    part of a matching run with the reference (difflib alignment)."""
    words = [(w, seg) for seg in segments for w in seg["words"] if normalize(w["text"])]
    ref = normalize(reference).split()
    matcher = difflib.SequenceMatcher(None, [normalize(w["text"]) for w, _ in words], ref, autojunk=False)
    correct = set()
    for block in matcher.get_matching_blocks():
        correct.update(range(block.a, block.a + block.size))
    return [(w, seg, i in correct) for i, (w, seg) in enumerate(words)]


def flag_quality(segments: list[dict], reference: str) -> dict:
    """How the wrong words relate to the segments flagged for review."""
    rows = word_correctness(segments, reference)
    flagged = [LEVELS.get(review_priority(seg["flags"]) or "", 0) >= LEVELS["medium"] for _, seg, _ in rows]
    wrong = [i for i, (_, _, ok) in enumerate(rows) if not ok]
    right = [i for i, (_, _, ok) in enumerate(rows) if ok]
    return {
        "words": len(rows),
        "wrong_words": len(wrong),
        "wrong_words_flagged": sum(flagged[i] for i in wrong),
        "words_flagged": sum(flagged),
        "right_words_flagged": sum(flagged[i] for i in right),
    }


def measure(service: EngineService, project_id: str, job: dict, item) -> dict:
    transcript = repo.get_transcript(service.db, project_id, include_words=True)
    segments = transcript["segments"]
    text = " ".join(s["text"] for s in segments)
    out: dict = {"status": job["status"], "segments": len(segments)}
    effective = job.get("effective") or {}
    for key in ("quality", "recheck", "validation"):
        if key in effective:
            out[key] = effective[key]
    if item.reference is not None:
        if item.no_speech:
            out["inserted_words"] = len(normalize(text).split())
        else:
            wer = word_error_rate(item.reference, text)
            cer = character_error_rate(item.reference, text)
            out.update(wer=round(wer.rate, 4), wer_errors=wer.errors, reference_words=wer.reference_length,
                       substitutions=wer.substitutions, deletions=wer.deletions, insertions=wer.insertions,
                       cer=round(cer.rate, 4))
            out["flags"] = flag_quality(segments, item.reference)
            out["word_details"] = [
                {"text": w["text"].strip(), "probability": w.get("probability"), "correct": ok,
                 "timing_source": w.get("timing_source"), "segment_priority": review_priority(seg["flags"]),
                 "segment_flags": [f["type"] for f in seg["flags"]]}
                for w, seg, ok in word_correctness(segments, item.reference)
            ]
    if item.turns is not None and transcript.get("speakers"):
        out["der"] = round(diarization_error_rate(item.turns, speaker_spans(transcript), collar=0.25).der, 4)
        out["speakers_found"] = len(transcript["speakers"])
    if item.words is not None:
        timing = timing_errors([w for s in segments for w in s["words"]], item.words)
        if timing:
            out["timing"] = timing
    out["text"] = text
    return out


def run_item(item, profile: str, args, root: Path, tag: str | None = None, **extra) -> dict:
    data = root / f"{tag or profile}-{item.id}"
    shutil.rmtree(data, ignore_errors=True)
    service = make_service(data, args)
    try:
        project = service.projects_create({"path": str(item.wav), "title": item.id})
        started = time.monotonic()
        job = service.jobs_start({"project_id": project["id"], "options": job_options(args, item, profile, **extra)})
        job = wait(service, job["id"])
        elapsed = time.monotonic() - started
        result = measure(service, project["id"], job, item)
        result["seconds"] = round(elapsed, 1)
        if job["status"] != "completed":
            result["error"] = job.get("error")
        return result
    finally:
        service.shutdown()
        if not args.keep:
            shutil.rmtree(data, ignore_errors=True)


def run_interrupted(item, args, root: Path) -> dict:
    """Quit the engine mid-transcription, start it again, resume the job, and
    compare with the same job run without interruption (same 20 s chunks)."""
    reference = run_item(item, "single", args, root, chunk_seconds=20, tag="uninterrupted")
    data = root / f"interrupted-{item.id}"
    shutil.rmtree(data, ignore_errors=True)
    first = make_service(data, args)
    project = first.projects_create({"path": str(item.wav), "title": item.id})
    job = first.jobs_start({"project_id": project["id"], "options": job_options(args, item, "single", chunk_seconds=20)})
    wait(first, job["id"], predicate=lambda j: any(
        s["stage"] == "transcribe" and s["status"] == "running" and (s.get("progress") or {}).get("done", 0) > 0
        for s in j["stages"]))
    first.shutdown()  # the app quits
    second = make_service(data, args)
    try:
        stopped = repo.get_job(second.db, job["id"])["status"]
        second.jobs.resume(job["id"])
        job = wait(second, job["id"])
        result = measure(second, project["id"], job, item)
        result["status_after_quit"] = stopped
        result["attempts"] = job["attempt"]
        result["uninterrupted_wer"] = reference.get("wer")
        result["uninterrupted_text_identical"] = reference.get("text") == result.get("text")
        return result
    finally:
        second.shutdown()
        if not args.keep:
            shutil.rmtree(data, ignore_errors=True)


# ----------------------------------------------------------------------------- report


def _percent(value) -> str:
    return "—" if value is None else f"{100 * value:.1f} %"


def markdown(results: dict, items: list) -> str:
    profiles = list(next(iter(results.values())).keys()) if results else []
    header = ["Item", "Categories", *[f"WER {p}" for p in profiles], "CER", "DER", "Review (high/medium)", "Notes"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for item in items:
        if item.id not in results:
            continue
        row = results[item.id]
        last = row[profiles[-1]]
        wers = []
        for p in profiles:
            r = row.get(p, {})
            wers.append(f"{r['inserted_words']} words" if "inserted_words" in r else _percent(r.get("wer")))
        review = last.get("validation", {}).get("review", {})
        der = "—" if last.get("der") is None else f"{last['der']:.3f}"
        cells = [item.id, ", ".join(item.categories), *wers, _percent(last.get("cer")), der,
                 f"{review.get('high', 0)}/{review.get('medium', 0)}", item.notes]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def summarise(results: dict, items: list) -> dict:
    by_category: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for item in items:
        for profile, r in results.get(item.id, {}).items():
            if r.get("reference_words"):
                for cat in item.categories:
                    acc = by_category[cat][profile]
                    acc["errors"] += r["wer_errors"]
                    acc["words"] += r["reference_words"]
    out = {cat: {p: {"wer": round(v["errors"] / v["words"], 4), "words": v["words"]} for p, v in profs.items()}
           for cat, profs in by_category.items()}
    flags = defaultdict(lambda: defaultdict(int))
    for item in items:
        for profile, r in results.get(item.id, {}).items():
            for k, v in (r.get("flags") or {}).items():
                flags[profile][k] += v
    return {"wer_by_category": out, "review_flags_vs_errors": {p: dict(v) for p, v in flags.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--whisper", type=Path, required=True)
    parser.add_argument("--backend", default="faster-whisper")
    parser.add_argument("--mode", default="balanced", help="decoding preset (beam size etc.)")
    parser.add_argument("--diarization", type=Path)
    parser.add_argument("--alignment", type=Path)
    parser.add_argument("--speaker-corpus", type=Path)
    parser.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "verbatim-benchmark")
    parser.add_argument("--profiles", default="single,multi")
    parser.add_argument("--items", help="comma-separated item ids (default: all)")
    parser.add_argument("--long", action="store_true", help="include the long recording")
    parser.add_argument("--interrupted", action="store_true", help="add the interrupted-processing run")
    parser.add_argument("--force-language", action="store_true", help="give the known language instead of detecting it")
    parser.add_argument("--keep", action="store_true", help="keep the engine data folders")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    items = corpus_module.build(args.cache, speaker_corpus=args.speaker_corpus, long=args.long)
    if args.items:
        wanted = set(args.items.split(","))
        items = [i for i in items if i.id in wanted]
    profiles = [p for p in args.profiles.split(",") if p]
    root = args.cache / "runs"
    root.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, dict]] = {}
    for item in items:
        results[item.id] = {}
        for profile in profiles:
            r = run_item(item, profile, args, root)
            results[item.id][profile] = r
            print(f"{item.id:22} {profile:7} wer={r.get('wer')} der={r.get('der')} "
                  f"inserted={r.get('inserted_words')} recheck={ {k: v for k, v in (r.get('recheck') or {}).items() if k in ('replaced', 'recovered', 'unresolved', 'no_acoustic_support', 'confirmed')} } "
                  f"{r['seconds']}s", flush=True)
    interrupted = None
    if args.interrupted:
        item = next(i for i in items if i.id == "conversation-2") if any(i.id == "conversation-2" for i in items) else None
        if item is not None:
            interrupted = {"item": item.id, **run_interrupted(item, args, root)}
            print("interrupted:", {k: interrupted.get(k) for k in (
                "status_after_quit", "attempts", "wer", "uninterrupted_wer", "uninterrupted_text_identical")})
    report = {"model": str(args.whisper), "backend": args.backend, "mode": args.mode, "profiles": profiles,
              "items": {i.id: {"categories": i.categories, "real": i.real, "notes": i.notes, "language": i.language}
                        for i in items},
              "results": results, "summary": summarise(results, items), "interrupted": interrupted}
    print(json.dumps(report["summary"], indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    if args.markdown:
        args.markdown.write_text(markdown(results, items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
