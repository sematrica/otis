# Verbatim — Architecture

Status: **Stage 1 (Foundation)**. This document records the selected libraries,
their verified compatibility, model requirements, and known risks. Anything
not yet verified is marked as such.

## 1. Goals that drive the design

1. Accuracy over speed. Never paraphrase, never invent words, preserve uncertainty.
2. Local-first. After models are installed, local media is processed offline.
3. Replaceable AI components. Each model sits behind a narrow, capability-aware
   interface so it can be swapped without touching the UI or storage.
4. Long recordings (multi-hour) on a 16 GB Apple Silicon machine: stream audio
   from disk, chunk, persist intermediate results, recover after interruption.
5. Launches from Finder. No Terminal, no manually activated Python environment.

## 2. Process model

```
┌──────────────────────────── Verbatim.app (Swift 6 / SwiftUI / AVKit) ───────────────────────────┐
│  Views ── AppModel (@MainActor) ── EngineClient (actor) ── JSON-lines over stdin/stdout ──┐     │
└──────────────────────────────────────────────────────────────────────────────────────────┼─────┘
                                                                                           ▼
                    verbatim-engine serve   (Python; long-lived; imports no ML libraries)
                      ├─ SQLite database (WAL)      projects · jobs · stages · chunks · segments · words
                      ├─ Model Manager             catalog, explicit downloads, deletion, storage location
                      ├─ Export Manager            TXT (Stage 1)
                      └─ Job Manager ──spawns──▶ verbatim-engine worker --job <id>   (one at a time)
                                                   extract → plan → load model → transcribe chunks → finalize
                                                   (exits after each job, releasing all model memory)
```

Why a separate worker process per job:

* **Memory.** On a 16 GB machine large models must never co-reside. Exiting the
  worker is the only reliable way to return all memory held by CTranslate2/MLX.
* **Cancellation.** Inference calls cannot be interrupted mid-kernel. Terminating
  the worker is immediate; completed chunks are already persisted.
* **Isolation.** A native crash in an ML library fails one job, not the app's
  data access. The server marks the job failed/interrupted and stays up.
* The worker watches its parent PID and exits if the server disappears, so no
  orphaned inference keeps running after the app quits.

Why the Python engine (not Swift) owns SQLite: one schema and one migration path,
fully testable on any CI host; the Swift app talks to it through a typed protocol
(`docs/ENGINE_PROTOCOL.md`). Both server and worker write with WAL mode and a busy
timeout; writes are short transactions.

## 3. Module map (the ten components)

| # | Component | Stage | Location |
|---|-----------|-------|----------|
| 1 | Media Import Manager | 1 | `engine/src/verbatim_engine/media/` (probe + validation), `app/Sources/Verbatim/Views/ImportView.swift` |
| 2 | Audio Processing Engine | 1 | `engine/src/verbatim_engine/audio/` (decode → 16 kHz mono WAV, chunk planner, WAV reader) |
| 3 | Speech Recognition Engine | 1 | `engine/src/verbatim_engine/asr/` (backend protocol, faster-whisper, mlx-whisper, merge) |
| 4 | Speaker Diarization Engine | 2 | `engine/src/verbatim_engine/diarization/` (planned) |
| 5 | Word Alignment Engine | 2 | `engine/src/verbatim_engine/alignment/` (planned) |
| 6 | Transcript Verification Engine | 3 | `engine/src/verbatim_engine/verification/` (planned) |
| 7 | Transcript Editor | 2+ | Swift (Stage 1 ships a read-only transcript view with click-to-seek) |
| 8 | Export Manager | 1 (TXT) / 4 | `engine/src/verbatim_engine/export/` |
| 9 | Model Manager | 1 (basic) / 5 | `engine/src/verbatim_engine/models/` |
| 10 | Persistent Job Manager | 1 | `engine/src/verbatim_engine/jobs/` |

## 4. Selected libraries and verified compatibility

Verified = installed and exercised in this repository's tests (Linux x86_64
dev container and/or the macOS arm64 CI runner). "CI" items are verified only on
the GitHub-hosted macOS runner because the dev container cannot reach them.

| Concern | Library | Version pin | Why | Verified |
|---------|---------|-------------|-----|----------|
| UI | SwiftUI, AVKit, AppKit | macOS 13 SDK+ | native; `NavigationSplitView`, `VideoPlayer` are macOS 13-compatible; `ObservableObject` used instead of `@Observable` (macOS 14+) | CI build |
| Language | Swift 6 language mode, SwiftPM | tools 6.0 | strict concurrency | CI build |
| Decode / resample | **PyAV** (FFmpeg 7 libraries, in-process) | `av>=15,<16` | No shell commands; no `ffmpeg` binary on `PATH` needed (Finder-launched apps don't inherit Homebrew `PATH`); streaming decode keeps memory flat. **PyAV 16+ ships macOS 14+ wheels only**, 15.x still ships macOS 13 arm64 wheels. | Linux + CI |
| ASR (CPU, portable) | **faster-whisper** on CTranslate2 | `faster-whisper==1.2.1`, `ctranslate2>=4.4,<5` | Beam search, temperature fallback, Silero VAD (bundled ONNX, no download), word timestamps via cross-attention DTW, int8/float32 on Apple Silicon CPU. CTranslate2 has **no Metal backend** on macOS. | Linux (real weights) + CI |
| ASR (Apple GPU) | **mlx-whisper** on MLX | `mlx-whisper==0.4.3` | Metal acceleration on Apple Silicon. **Greedy decoding only — no beam search**. Word timestamps supported. MLX ≥0.30 ships macOS 14+ wheels only, so the MLX backend is unavailable on macOS 13 and the engine reports it as such. Its declared PyTorch dependency is excluded (only its checkpoint converter imports torch). | CI (Apple M1, Metal GPU) |
| VAD | Silero VAD v6 via faster-whisper | bundled | Reduces hallucinated text in silence; used directly by faster-whisper and as `clip_timestamps` for mlx-whisper | Linux |
| ONNX runtime | onnxruntime | `>=1.17,<1.24` | needed by Silero VAD; 1.24+ is macOS 14+ only | Linux |
| Model download | huggingface_hub | `>=0.23` | explicit `snapshot_download` into a plain directory; `local_files_only` when loading | CI |
| Storage | SQLite (Python `sqlite3`) | system | WAL, foreign keys, migrations; FTS5 planned for Stage 4 | Linux + CI |
| Diarization (Stage 2) | pyannote.audio | TBD | see risks | not yet |
| Alignment (Stage 2) | CTC forced alignment (wav2vec2 family) | TBD | see risks | not yet |
| YouTube (Stage 4) | yt-dlp | TBD | authorized content only; captions when available | not yet |

Python runtime: CPython 3.12 (python-build-standalone, relocatable) embedded in
the app bundle by `scripts/build_app.sh`.

### Capability model

Every ASR backend reports what it can actually do, and the job records the
*effective* settings it ran with (never the requested ones):

| Capability | faster-whisper | mlx-whisper |
|------------|----------------|-------------|
| Beam search | yes | **no** (greedy + temperature fallback) |
| Word timestamps | yes (DTW on alignment heads) | yes (DTW) |
| Built-in VAD | yes (Silero) | no (engine supplies `clip_timestamps` from Silero) |
| Language detection | yes | yes |
| Language reported per | chunk (not exposed per segment) | chunk |
| Acceleration on Apple Silicon | CPU (Accelerate / int8) | Metal GPU |
| Measurable progress | yes (segment end times) | yes (decoded frames) |

## 5. Pipeline (Stage 1)

1. **Prepare** — re-validate the source (exists, regular file, size/mtime unchanged).
2. **Extract** — PyAV decodes the best audio stream and resamples to 16 kHz mono
   s16 PCM, streaming into a WAV in the job's working directory. Progress is
   decoded time ÷ container duration (only when the duration is known).
3. **Plan** — one streaming pass computes 20 ms RMS energy; for recordings longer
   than ~12 min, cut points are placed at the quietest 0.5 s window within ±30 s
   of each 10-minute boundary. When the cut lies in a pause, each chunk's audio
   extends into the pause but never past it, so neighbouring chunks never hear
   the same speech. Only when no pause exists near a boundary is the audio padded
   by 1 s on both sides (overlap). The first chunk gets 0.5 s of silent lead-in so
   a recording that starts mid-word keeps its first word.
4. **Load model** — one model, loaded once per job, from local files only.
5. **Transcribe** — chunks are processed sequentially; the tail of the previous
   chunk's text is passed as the initial prompt to preserve context. Each chunk's
   raw result is persisted before moving on (resume point).
6. **Finalize** — at pause cuts, chunk results are concatenated unchanged (a test
   asserts the transcript equals the concatenation of every chunk's raw words). At
   cuts inside speech, the two decodings of the overlap are aligned and spliced at
   a word both agree on; if they disagree, a loss-free time rule plus a
   conservative duplicate check applies. Every such decision is recorded in the
   segment's `flags`. Segments, words, confidence statistics and detected language
   are written in one transaction.

Hallucination guards already in Stage 1: VAD, temperature fallback with
compression-ratio / log-probability / no-speech thresholds, prompt reset on
high temperature. Per-segment `avg_logprob`, `no_speech_prob`,
`compression_ratio`, and `temperature` are stored for Stage 3 quality analysis.

Progress reporting never invents numbers: a stage reports a fraction only when it
has a real numerator and denominator; otherwise the UI shows an indeterminate
state with elapsed time. ETA is derived from the measured processing rate and is
shown only after enough work has been observed.

## 6. Processing modes

| Mode | Model | Decoding | Stage 1 behaviour |
|------|-------|----------|-------------------|
| Fast | Whisper Small | greedy (beam 1) | single pass |
| Balanced | Whisper Large-v3 Turbo | beam 5 where supported | single pass |
| Maximum Accuracy | Whisper Large-v3 | beam 5 / best-of 5 where supported; float32 on CPU | single pass. Multi-pass verification (passes 5–8) arrives in Stage 3; the UI says so. |

Whisper Medium is available for manual selection.

## 7. Model requirements

Download sizes were verified against the HuggingFace API by
`scripts/verify_model_catalog.py` in CI (all eight entries exist, contain the
required files, and match the sizes shown to users). Peak memory is an estimate
for a single loaded model plus decoding buffers and is **not yet measured**.

| Model | Parameters | Download (fp16) | Est. peak memory | Backends |
|-------|-----------:|----------------:|-----------------:|----------|
| Whisper Small | 244 M | ~0.5 GB | ~1–2 GB | faster-whisper, mlx |
| Whisper Medium | 769 M | ~1.5 GB | ~2–3 GB | faster-whisper, mlx |
| Whisper Large-v3 | 1550 M | ~3.1 GB | ~4–7 GB (float32 CPU is the upper bound) | faster-whisper, mlx |
| Whisper Large-v3 Turbo | 809 M | ~1.6 GB | ~2–4 GB | faster-whisper, mlx |

Models are **never** downloaded implicitly. The user sees the size and confirms;
model loading always uses local files only. The storage folder is configurable
(external SSD supported).

## 8. Security and privacy

* No network access during transcription. Downloads happen only on explicit user
  action. No telemetry.
* No shell: every subprocess (`Process` in Swift, `subprocess` in Python) is
  launched with an argument vector. Media is decoded in-process through PyAV.
* Input validation: regular files only, symlinks resolved, size/mtime recorded,
  container probed before a project is created; paths are never interpolated
  into commands or SQL (parameterized queries only).
* All working files live under the app's data directory
  (`~/Library/Application Support/Verbatim`), per job, and are removed with the
  project.
* API keys (Stage 3 LLM providers) will be stored in the macOS Keychain.

## 9. Known risks and open questions

1. **No macOS toolchain in the development container.** The container is Linux
   x86_64 without Swift; Swift code is compiled and tested only on the
   GitHub-hosted macOS arm64 runner. SwiftUI behaviour (layout, drag-and-drop,
   playback) has no automated UI tests in Stage 1.
2. **HuggingFace is unreachable from the development container.** Local
   end-to-end tests use real Whisper tiny/base/small CTranslate2 weights from a
   GitHub-hosted mirror, with a tokenizer rebuilt from OpenAI's tiktoken ranks and
   verified token-by-token. CI downloads the official models through the engine's
   own download path and verifies the catalog.
3. **macOS 13 support is partial**: the MLX backend requires macOS 14+ with
   current wheels; on macOS 13 only the CPU backend (faster-whisper) is available.
4. **Large-v3 on CPU is slow.** faster-whisper has no Metal backend; on an M2 Pro
   the MLX backend is the practical choice for large models, but it lacks beam search.
5. **FFmpeg licensing.** The Linux PyAV 15 wheel's FFmpeg build enables libx264/
   libx265 (GPL) in addition to LGPLv3 components. Redistributing the app with
   PyAV's prebuilt macOS wheel may impose GPL obligations; Stage 5 must verify the
   macOS wheel's build configuration (the packaging script records it) or rebuild
   PyAV against an LGPL-only FFmpeg.
6. **Bundle size.** The Stage 1 bundle is about 800 MB (MLX, numba/llvmlite,
   scipy, onnxruntime, CTranslate2, FFmpeg). PyTorch is already excluded and
   MLX transcription is verified without it; further trimming is Stage 5 work.
7. **pyannote.audio (Stage 2)**: current pipelines are gated on HuggingFace (the
   user must accept terms and provide a token once); PyTorch MPS support and memory
   use on 16 GB must be measured; licence terms of each pipeline must be checked.
8. **Forced alignment (Stage 2)**: wav2vec2 CTC aligners cover a limited set of
   languages, and `torchaudio`'s alignment APIs have been deprecated/removed in
   recent releases; a self-contained CTC Viterbi implementation may be needed.
9. **YouTube (Stage 4)**: yt-dlp breaks when YouTube changes; current yt-dlp needs
   an external JavaScript runtime for YouTube; terms of service restrict use to
   content the user is authorized to download. No DRM or authentication bypass.
10. **Code signing & notarization (Stage 5)**: every embedded `.so`/`.dylib` must be
    signed with the hardened runtime; numba/llvmlite and MLX may need JIT-related
    entitlements. The Stage 1 bundle is ad-hoc signed and not notarized.
11. **Memory pressure**: Stage 2 adds diarization and alignment models. The worker
    must run them sequentially and release each before loading the next.
