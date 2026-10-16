#!/usr/bin/env python3
"""Download the real-speech speaker corpus used by the diarization tests.

All files come from the pyannote-audio repository (MIT licence) at a pinned
commit and are checked against SHA-256 checksums:

* ``sample.wav`` / ``sample.rttm`` - a 30 s two-person telephone
  conversation with a reference speaker annotation (pyannote tutorial asset);
* ``dev*/trn*/tst*.wav`` and ``debug.*.rttm`` - 30 s excerpts of AMI meeting
  recordings (AMI corpus, CC BY 4.0) with 1 to 4 speakers and heavy overlap,
  used by pyannote's own tests.

    python scripts/fetch_speaker_corpus.py OUTPUT_DIR

Tests find the corpus through ``VERBATIM_TEST_SPEAKER_CORPUS=OUTPUT_DIR``.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import httpx2  # the engine's HTTP client (system trust store)

COMMIT = "78a6e44b6b3905eb51e458879857dd82d0f9ca96"
BASE = f"https://raw.githubusercontent.com/pyannote/pyannote-audio/{COMMIT}"

FILES = {
    "tutorials/assets/sample.wav": "c319b4abca767b124e41432d364fd7df006cb26bb79d09326c487d606a134e6e",
    "tutorials/assets/sample.rttm": "d78fe62c69d8e6dcbb42c26adfce83faccb374c5a1e6d987fe37f85f1c173c87",
    "tests/data/debug.development.rttm": "9667a672072f7d0ad5b647cfaed987d8c52209a4e1ddc252fed06d0b955e7aeb",
    "tests/data/debug.train.rttm": "972ee04414fc7644f26eabe1909d6a0f6a3fa57ab840719ec0112d26f3a957d5",
    "tests/data/debug.test.rttm": "eae147d74d3f08002a5028183538eb35e6a7c7a77fe7d87c51eec95852d33161",
    "tests/data/dev00.wav": "187386cda67317574da5b30068a8b853e52cc6df0d13abfec381912dfa9e7461",
    "tests/data/dev01.wav": "13834217e1bd46a581af1b8040018bfba1e65ce1e0b9ee46cdcefa9334b9ddea",
    "tests/data/trn01.wav": "b5e8d91105604d25cb05cc3840d015d88bd6689a2acba35506dd0d7dcb00668a",
    "tests/data/trn02.wav": "2e4ce968332e68cac5ed74cd00f96b63ed6c034667583be34a32a7c2aa260803",
    "tests/data/trn03.wav": "d12d132a6d4788f85efaa8e18563e575746e6a6fcc928e919c7c0e6a3e0238b2",
    "tests/data/trn04.wav": "afa4bd13ee05be34953beb594d0d755ba83284c5a742ae27e388952d41c4d0f1",
    "tests/data/trn05.wav": "ec694aa5c9eea7453263eaf90d38ed921c1b6f8cc4f52897d9a4457a504d668d",
    "tests/data/trn06.wav": "e70cbed4f13c7971fdf9d424b079d87a25ed1ce3928da0bccd18f143eb4ac359",
    "tests/data/trn07.wav": "77628cd91dcc27c61f9b709e670f003cb860d597fda56e7ccb0ccbbc42d5a26f",
    "tests/data/trn08.wav": "a7d3397e82cb1025307f08b0fe8459a9d43fe61f3d9ce674f17819c7450bfac5",
    "tests/data/trn09.wav": "8e18bc410e6d54a166bbba2ff0b46a222626706788fa222add4f97bca842f822",
    "tests/data/tst01.wav": "a40925742870d05363eb781bd2cf1501fc37883cf003389175a39b24a5b2c7c9",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for rel, digest in FILES.items():
        target = out / Path(rel).name
        if target.is_file() and sha256(target) == digest:
            continue
        response = httpx2.get(f"{BASE}/{rel}", timeout=60, follow_redirects=True)
        response.raise_for_status()
        data = response.content
        if hashlib.sha256(data).hexdigest() != digest:
            print(f"checksum mismatch for {rel}", file=sys.stderr)
            return 1
        target.write_bytes(data)
        print(f"fetched {target.name} ({len(data)} bytes)")
    print(f"speaker corpus ready in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
