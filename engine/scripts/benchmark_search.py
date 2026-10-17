#!/usr/bin/env python3
"""Measure full-text search speed on a large synthetic library.

Builds a library of PROJECTS × PASSAGES transcripts through the engine's own
storage (the FTS5 index is filled by its triggers, as in the app), with
6–18 words per passage drawn from a Zipf-distributed vocabulary of made-up
words, then times ``search.transcripts`` for typical queries. The text is
not language: these numbers describe the index and the query path, not
relevance.

    python scripts/benchmark_search.py --data-dir /tmp/search-bench \
        --projects 100 --passages 2000 [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

from verbatim_engine.paths import EnginePaths  # noqa: E402
from verbatim_engine.rpc.service import EngineService  # noqa: E402
from verbatim_engine.storage import repo  # noqa: E402
from verbatim_engine.transcript.model import TSegment, TWord  # noqa: E402

SYLLABLES = ["ka", "lo", "mi", "ter", "san", "vo", "ri", "del", "pa", "nu", "gor", "e", "is", "tan", "bru"]
RUNS = 7


def build(service: EngineService, media: Path, projects: int, passages: int,
          rng: random.Random) -> tuple[list[str], str]:
    """The vocabulary (most frequent first) and the last project's id."""
    vocabulary = sorted({"".join(rng.choice(SYLLABLES) for _ in range(rng.randint(1, 4))) for _ in range(20000)})
    rng.shuffle(vocabulary)
    weights = [1 / (rank + 1) for rank in range(len(vocabulary))]
    last = None
    for p in range(projects):
        last = service.projects_create({"path": str(media), "title": f"Project {p}"})["id"]
        segments, t = [], 0.0
        for _ in range(passages):
            words = rng.choices(vocabulary, weights, k=rng.randint(6, 18))
            timed = [TWord(t + 0.3 * i, t + 0.3 * i + 0.25, " " + w, 0.9) for i, w in enumerate(words)]
            segments.append(TSegment(timed[0].start, timed[-1].end, " ".join(words) + ".", timed, "en"))
            t += 0.3 * len(words) + 0.5
        repo.replace_transcript(service.db, last, None, segments, "en", [])
    return vocabulary, last


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="an empty folder for the library")
    parser.add_argument("--projects", type=int, default=100)
    parser.add_argument("--passages", type=int, default=2000, help="passages per project")
    parser.add_argument("--media", type=Path, default=HERE.parent / "tests" / "corpus" / "jfk.wav",
                        help="any audio file (projects need a real source)")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.data_dir.exists() and any(args.data_dir.iterdir()):
        parser.error("--data-dir must be empty or not exist")

    service = EngineService(EnginePaths.create(args.data_dir), lambda *_: None)
    service.start()
    try:
        started = time.monotonic()
        vocabulary, last = build(service, args.media, args.projects, args.passages, random.Random(0))
        built = time.monotonic() - started
        size = sum(f.stat().st_size for f in args.data_dir.glob(service.paths.db_path.name + "*"))
        queries = [
            ("rare word", {"query": vocabulary[-1]}),
            ("mid-frequency word", {"query": vocabulary[200]}),
            ("two words", {"query": f"{vocabulary[3]} {vocabulary[40]}"}),
            ("quoted phrase", {"query": f'"{vocabulary[1]} {vocabulary[2]}"'}),
            ("three-letter prefix", {"query": vocabulary[50][:3]}),
            ("most common word, one project", {"query": vocabulary[0], "project_id": last}),
            ("most common word", {"query": vocabulary[0]}),
        ]
        search = service.methods()["search.transcripts"]
        report = {"projects": args.projects, "passages": args.projects * args.passages,
                  "build_seconds": round(built, 1), "database_bytes": size, "queries": []}
        print(f"{report['passages']} passages in {args.projects} projects; built in {built:.0f} s; "
              f"database {size / 1e6:.0f} MB")
        print("| Query | Results | Median | Max |\n|---|---|---|---|")
        for name, params in queries:
            times = []
            for _ in range(RUNS):
                tick = time.perf_counter()
                found = search(params)
                times.append(time.perf_counter() - tick)
            results = f"{len(found['results'])}{' (limit)' if found['truncated'] else ''}"
            print(f"| {name} | {results} | {statistics.median(times) * 1000:.1f} ms | {max(times) * 1000:.1f} ms |")
            report["queries"].append({"name": name, "query": params["query"], "results": len(found["results"]),
                                      "truncated": found["truncated"],
                                      "median_ms": round(statistics.median(times) * 1000, 2),
                                      "max_ms": round(max(times) * 1000, 2)})
    finally:
        service.shutdown()
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
