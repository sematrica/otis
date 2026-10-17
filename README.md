# Verbatim

Accurate, local transcription of videos and podcasts for macOS on Apple
Silicon. A native SwiftUI app drives a Python engine that runs Whisper
(faster-whisper on the CPU, or MLX on the Apple GPU) entirely on your Mac.

**Status: Stage 4 (Media import, search, exports, queue).** Everything from
Stages 1–3 (local media import, real Whisper transcription, persistent jobs
with cancel/resume/crash recovery, speakers, word alignment, the three-panel
workspace, corrections with undo/redo, quality analysis, acoustic re-checking
of doubtful passages, validation, optional AI review) plus:

* **Authorized YouTube import** — paste a link of a video you may download
  (your own, licensed for reuse, or with the rights holder's permission).
  Verbatim shows the title, channel, licence, length and published captions,
  asks you to confirm, retrieves the audio (and video up to 720p, if you want
  it), then transcribes it like a local file. It never signs in, uses no
  cookies and does not bypass DRM or restrictions; when YouTube refuses, it
  says why, and you can import a file instead. Ads YouTube inserts during
  playback are not part of the retrieved file and are not transcribed;
  sponsor messages inside the video are, and passages in sponsor chapters
  are labelled.
* **Published captions** are kept alongside the transcript as their own
  source (exportable as SRT/VTT) and compared with it: passages where they
  clearly differ are flagged for you to listen to. They never replace
  transcribed words.
* **Full-text search** across all transcripts (library) or in the open one
  (⌘F): words match as prefixes, "quoted phrases" match exactly, results
  jump to the passage.
* **Every export format** — TXT, SRT, WebVTT, JSON (metadata, speaker IDs and
  names, segment and word times, corrected and original transcript,
  uncertainty flags), CSV (per passage or per word) and Markdown.
* **Persistent processing queue** — reorder waiting jobs, pause after the
  current job or immediately (it resumes where it stopped), and interrupted
  work is resumed when Verbatim starts again.

Production hardening (notarization, updates, bundle size) is Stage 5; see
`docs/ARCHITECTURE.md` and `docs/STAGE4_REPORT.md`.

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
uv run verbatim-engine transcribe ~/talk.mp4 --format srt -o talk.srt          # txt srt vtt json csv md
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
uv run python scripts/evaluate_speaker_review.py --whisper … --diarization … \
    --alignment … --corpus /tmp/speakers                       # do review flags find wrong speakers?
uv run python scripts/benchmark_accuracy.py --whisper … --diarization … --alignment … \
    --speaker-corpus /tmp/speakers --interrupted --markdown results.md   # Stage 3 accuracy benchmark
VERBATIM_TEST_OLLAMA_MODEL=qwen2.5:0.5b uv run pytest tests/test_stage3_llm.py   # AI review, local Ollama
VERBATIM_TEST_MODEL_DIR=… VERBATIM_TEST_YOUTUBE_URL=https://www.youtube.com/watch?v=… \
    uv run pytest tests/test_stage4_models.py -k real_youtube -s   # YouTube itself (media only if CC or
                                                                   # VERBATIM_TEST_YOUTUBE_AUTHORIZED=1)
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
engine/    Python engine: media, YouTube import, audio, ASR, diarization, alignment, verification, AI review, transcript corrections, search, exports, jobs and queue, storage, RPC, CLI
shared/    protocol fixtures captured from a real engine run (decoded by Swift tests)
scripts/   build_app.sh, smoke_test_app.sh, app_self_test.sh, dev_run.sh, verify_model_catalog.py
docs/      ARCHITECTURE.md, ENGINE_PROTOCOL.md, STAGE1_REPORT.md … STAGE4_REPORT.md (results and limitations)
```
