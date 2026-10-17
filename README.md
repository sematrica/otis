# Verbatim

Accurate, local transcription of videos and podcasts for macOS on Apple
Silicon. A native SwiftUI app drives a Python engine that runs Whisper
(faster-whisper on the CPU, or MLX on the Apple GPU) entirely on your Mac.

**Status: Stage 3 (Verification and accuracy).** Everything from Stages 1–2
(local media import, real Whisper transcription, persistent jobs with
cancel/resume/crash recovery, speakers, word alignment, the three-panel
workspace, corrections with undo/redo, TXT export) plus:

* **Quality analysis on every transcript** — unclear speech, possible
  hallucinations, repetitions and speech that no word covers are flagged with a
  severity; the Review menu lists the passages that need a look, most severe
  first, and uncertain words are coloured.
* **Maximum Accuracy re-checks doubtful passages** — they are transcribed again
  from the original recording with context and other settings, and the
  wordings are compared by acoustic evidence (an independent CTC model scores
  each wording against the audio). A wording replaces the first pass only with
  a clear margin; close calls are kept as alternatives for you to choose; words
  are never removed automatically. Any passage can also be re-checked on demand.
* **Validation** — timestamps, word order, duplicates and untranscribed speech
  are checked on the final transcript; unresolved doubts stay visible.
* **Optional AI review** — a local model (Ollama) or, with your explicit
  permission, OpenAI can flag suspicious passages, suggest punctuation, point
  out inconsistent spellings and suggest passages to re-check. Nothing changes
  until you accept a suggestion; every request and answer is recorded, and the
  OpenAI key stays in the Keychain.
* **Accuracy benchmark** — `engine/scripts/benchmark_accuracy.py` measures WER,
  CER, DER, word-time error and flag quality on a reproducible corpus (real and
  synthetic speech), with and without re-checking; CI runs it.

YouTube import, the other export formats and search are later stages; see
`docs/ARCHITECTURE.md` and `docs/STAGE3_REPORT.md`.

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
uv run python scripts/evaluate_speaker_review.py --whisper … --diarization … \
    --alignment … --corpus /tmp/speakers                       # do review flags find wrong speakers?
uv run python scripts/benchmark_accuracy.py --whisper … --diarization … --alignment … \
    --speaker-corpus /tmp/speakers --interrupted --markdown results.md   # Stage 3 accuracy benchmark
VERBATIM_TEST_OLLAMA_MODEL=qwen2.5:0.5b uv run pytest tests/test_stage3_llm.py   # AI review, local Ollama
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
engine/    Python engine: media, audio, ASR, diarization, alignment, verification, AI review, transcript corrections, jobs, storage, RPC, CLI
shared/    protocol fixtures captured from a real engine run (decoded by Swift tests)
scripts/   build_app.sh, smoke_test_app.sh, app_self_test.sh, dev_run.sh, verify_model_catalog.py
docs/      ARCHITECTURE.md, ENGINE_PROTOCOL.md, STAGE1_REPORT.md … STAGE3_REPORT.md (results and limitations)
```
