# Verbatim

Accurate, local transcription of videos and podcasts for macOS on Apple
Silicon. A native SwiftUI app drives a Python engine that runs Whisper
(faster-whisper on the CPU, or MLX on the Apple GPU) entirely on your Mac.

**Status: Stage 2 (Speaker Intelligence).** Everything from Stage 1 (local
media import, real Whisper transcription, persistent projects and jobs with
cancel/resume/crash recovery, TXT export, explicit model management) plus:

* **Speakers** — who spoke when, found on your Mac (pyannote's segmentation and
  WeSpeaker models on ONNX Runtime): the number of speakers is estimated (or
  given), overlapping speech is detected, segments are split where the speaker
  changes, and doubtful attributions are flagged. Speakers stay anonymous
  ("Speaker 1") until you name them.
* **Word alignment** — precise word times by CTC forced alignment (NeMo
  Conformer-CTC for English, MMS for 1,100+ languages); Whisper's times are kept
  and flagged where alignment is rejected.
* **Three-panel workspace** — speakers | video | transcript; the playing segment
  and word are highlighted; clicking a segment or word seeks the video.
* **Corrections with undo/redo** — rename, merge and add speakers; reassign,
  split, merge and edit segments; mark passages uncertain. The AI-generated
  transcript is kept unchanged alongside.

Verification passes, YouTube import and the other export formats are later
stages; see `docs/ARCHITECTURE.md` and `docs/STAGE2_REPORT.md`.

## Requirements

* Apple Silicon Mac, macOS 13 or newer (the MLX GPU backend needs macOS 14+;
  macOS 13 uses the CPU backend).
* To build: Xcode 16 or newer (Swift 6) and [uv](https://docs.astral.sh/uv/).

## Build and launch

```sh
scripts/build_app.sh            # produces build/Verbatim.app
open build/Verbatim.app         # or double-click it in Finder
```

The bundle contains its own Python runtime and the engine; no Terminal setup
is needed to run it. On first use, open **Settings › Models** and download a
Whisper model, the speaker identification model (33 MB) and, for precise word
times, an alignment model (sizes and licences are shown before anything is
downloaded). An app built on your own Mac is ad-hoc signed and opens normally;
a build downloaded from CI is quarantined by Gatekeeper (it is not notarized
yet): on macOS 15 use System Settings › Privacy & Security › Open Anyway, or
`xattr -dr com.apple.quarantine Verbatim.app`.

Development loop without building a bundle:

```sh
scripts/dev_run.sh              # uv sync + swift run, using engine/.venv
```

## Command line

The engine also works headless:

```sh
cd engine && uv sync
uv run verbatim-engine models                                  # catalog and install state
uv run verbatim-engine download-model mlx/large-v3-turbo       # explicit download
uv run verbatim-engine download-model diarization/pyannote-wespeaker
uv run verbatim-engine download-model alignment/en-conformer-small
uv run verbatim-engine transcribe ~/talk.mp4 --mode maximum --num-speakers 2 -o talk.txt
```

## Tests

```sh
cd engine
uv run pytest -m "not model"                                   # no weights needed
uv run python scripts/fetch_speaker_corpus.py /tmp/speakers    # real speech + speaker annotations
VERBATIM_TEST_MODEL_DIR=/path/to/faster-whisper-small \
VERBATIM_TEST_DIARIZATION_DIR=/path/to/models/diarization/pyannote-wespeaker \
VERBATIM_TEST_ALIGNMENT_DIR=/path/to/models/alignment/en-conformer-small \
VERBATIM_TEST_SPEAKER_CORPUS=/tmp/speakers uv run pytest -m model
uv run python scripts/benchmark_word_timing.py --whisper … --aligner nemo:…   # word-time accuracy
cd ../app && swift test                                        # macOS only
```

Real-model tests run actual speech through real models (Whisper, diarization,
alignment) and measure WER, DER and word-time error; they are skipped (never
faked) when the models are not available. CI
(`.github/workflows/ci.yml`) runs everything on Linux and on an Apple Silicon
macOS runner, builds the app bundle and smoke-tests it.

## Layout

```
app/       Swift package: VerbatimCore (engine client, protocol types) and the SwiftUI app
engine/    Python engine: media, audio, ASR, diarization, alignment, transcript corrections, jobs, storage, RPC, CLI
shared/    protocol fixtures captured from a real engine run (decoded by Swift tests)
scripts/   build_app.sh, smoke_test_app.sh, app_self_test.sh, dev_run.sh, verify_model_catalog.py
docs/      ARCHITECTURE.md, ENGINE_PROTOCOL.md, STAGE1_REPORT.md, STAGE2_REPORT.md (results and limitations)
```
