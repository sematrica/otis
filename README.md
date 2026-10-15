# Verbatim

Accurate, local transcription of videos and podcasts for macOS on Apple
Silicon. A native SwiftUI app drives a Python engine that runs Whisper
(faster-whisper on the CPU, or MLX on the Apple GPU) entirely on your Mac.

**Status: Stage 1 (Foundation).** Local media import, real Whisper
transcription with timestamps, persistent projects and jobs (cancel, resume,
crash recovery), TXT export, and explicit model management. Speaker
diarization, word alignment, verification passes, YouTube import and the
other export formats are later stages; see `docs/ARCHITECTURE.md`.

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
model (sizes are shown before anything is downloaded). An app built on your
own Mac is ad-hoc signed and opens normally; a build downloaded from CI is
quarantined by Gatekeeper and must be opened with right-click › Open
(it is not notarized yet).

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
uv run verbatim-engine transcribe ~/talk.mp4 --mode maximum -o talk.txt
```

## Tests

```sh
cd engine
uv run pytest -m "not model"                                   # no weights needed
VERBATIM_TEST_MODEL_DIR=/path/to/faster-whisper-small uv run pytest -m model
cd ../app && swift test                                        # macOS only
```

Real-model tests transcribe actual speech with real Whisper weights; they are
skipped (never faked) when no weights are available. CI
(`.github/workflows/ci.yml`) runs everything on Linux and on an Apple Silicon
macOS runner, builds the app bundle and smoke-tests it.

## Layout

```
app/       Swift package: VerbatimCore (engine client, protocol types) and the SwiftUI app
engine/    Python engine: media, audio, ASR backends, jobs, storage, RPC server, CLI
shared/    protocol fixtures captured from a real engine run (decoded by Swift tests)
scripts/   build_app.sh, smoke_test_app.sh, dev_run.sh, verify_model_catalog.py
docs/      ARCHITECTURE.md, ENGINE_PROTOCOL.md
```
