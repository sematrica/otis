# Verbatim — Architecture

Status: **Stage 2 (Speaker Intelligence)** on top of Stage 1 (Foundation).
This document records the selected libraries, their verified compatibility,
model requirements, and known risks. Anything not yet verified is marked as
such. Measured results are in `docs/STAGE1_REPORT.md` and `docs/STAGE2_REPORT.md`.

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
                      ├─ SQLite database (WAL)      projects · jobs · stages · chunks · segments · words ·
                      │                              speakers · original transcripts · edit history
                      ├─ Model Manager             catalog, explicit downloads, deletion, storage location
                      ├─ Transcript Editor         corrections with undo/redo (engine side)
                      ├─ Export Manager            TXT with speaker turns
                      └─ Job Manager ──spawns──▶ verbatim-engine worker --job <id>   (one at a time)
                                                   extract → plan → load model → transcribe chunks →
                                                   (unload Whisper) → diarize → align → finalize
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
| 1 | Media Import Manager | 1 | `engine/src/verbatim_engine/media/` (probe + validation), `app/Sources/Verbatim/Views/LibraryView.swift` |
| 2 | Audio Processing Engine | 1 | `engine/src/verbatim_engine/audio/` (decode → 16 kHz mono WAV, chunk planner, WAV reader, level normalisation) |
| 3 | Speech Recognition Engine | 1 | `engine/src/verbatim_engine/asr/` (backend protocol, faster-whisper, mlx-whisper, merge) |
| 4 | Speaker Diarization Engine | 2 | `engine/src/verbatim_engine/diarization/` (segmentation, embeddings, clustering, timeline, word/segment assignment) |
| 5 | Word Alignment Engine | 2 | `engine/src/verbatim_engine/alignment/` (CTC models, Viterbi, aligner, boundary refinement) |
| 6 | Transcript Verification Engine | 3 | `engine/src/verbatim_engine/verification/` (planned) |
| 7 | Transcript Editor | 2 | `engine/src/verbatim_engine/transcript/` (data model, build, corrections + undo) and `app/Sources/Verbatim/Workspace/` |
| 8 | Export Manager | 1 (TXT) / 4 | `engine/src/verbatim_engine/export/` |
| 9 | Model Manager | 1–2 / 5 | `engine/src/verbatim_engine/models/` (HuggingFace and pinned-URL downloads) |
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
| Diarization | pyannote segmentation-3.0 + WeSpeaker ResNet34 (the models of pyannote's `speaker-diarization-3.1`), as ONNX exports, run on onnxruntime; pipeline re-implemented in NumPy/SciPy | pinned SHA-256 | Same models and algorithm as pyannote 3.1 without PyTorch (excluded from the bundle) and without gated HuggingFace downloads | Linux + CI (DER measured) |
| Clustering | SciPy (`linkage`, `linear_sum_assignment`) | `scipy>=1.13` | centroid-linkage agglomerative clustering; already in the bundle via mlx-whisper | Linux + CI |
| Speaker features | kaldi-native-fbank | `>=1.20` | Kaldi-compatible filterbanks (WeSpeaker and NeMo inputs), Apache-2.0 | Linux + CI |
| Alignment (English) | NVIDIA NeMo Conformer-CTC (Small, Medium), ONNX exports by sherpa-onnx | pinned SHA-256 | CC BY 4.0; testable offline; ~35 ms word-start error | Linux + CI |
| Alignment (multilingual) | Meta MMS-300M forced aligner (ONNX) | HF file, size-verified | 1,100+ languages; most precise measured (~18 ms); **CC BY-NC 4.0** | CI only (HF unreachable from the dev container) |
| Romanisation | uroman | `>=1.3` | non-Latin scripts → Latin letters for MMS (Apache-2.0) | Linux |
| CTC alignment | own NumPy Viterbi | — | `torchaudio.functional.forced_align` is deprecated and needs PyTorch | Linux + CI |
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

## 5. Pipeline

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

7. **Unload the recogniser**, then **diarize** — sliding 10 s windows (1 s step)
   through segmentation-3.0 give up to three local speakers per window (with
   overlaps); a WeSpeaker embedding of each local speaker's clean speech feeds
   centroid-linkage clustering with pyannote 3.1's tuned threshold, which
   estimates the number of speakers unless the user gives an exact number or a
   range. Local speakers are mapped to global ones and every window's opinion is
   combined into a frame-level timeline (16.9 ms frames) holding, per frame, the
   number of speakers, each speaker's activation (window agreement) and the
   embedding margin. The timeline is saved in the job folder (resume).
8. **Align** — for each group of segments (≤ 30 s), CTC emissions are computed
   on level-normalised audio and the transcript tokens are forced through them
   (Viterbi) with a wildcard token at the edges, between segments and in place of
   words the model cannot spell. Token spikes become word boundaries using pauses
   (energy) or, in connected speech, a per-model split of the gap. Segments whose
   alignment is implausible keep Whisper's times and are flagged. Results are
   appended per group to `alignment.jsonl` (resume).
9. **Finalize** — each word gets the speaker active for most of its duration;
   segments are split where the speaker changes; doubtful attributions are
   flagged; speakers are numbered by first appearance; the transcript, its
   immutable original copy and the speakers are written in one transaction.

### Speaker assignment thresholds (heuristics, not calibrated probabilities)

| Flag | Rule |
|------|------|
| `speaker_uncertain` | coverage of the assigned speaker < 0.5, window agreement < 0.5, embedding margin < 0.1 (only with 2+ speakers), or no voice under the words (nearest speaker within 1 s used) |
| `speaker_change_uncertain` | a word next to a change is < 75% covered by its speaker, overlaps, or was folded |
| `overlapping_speech` | ≥ 20% of the segment's word time, or ≥ 0.3 s, has 2+ speakers |
| `alignment_failed` | mean token log-probability < −4, median start shift vs. Whisper > 1.5 s, or a word longer than 4 s |

A run of words that switches speaker inside a segment and back again is
folded into the surrounding speaker when its own coverage is < 0.5.

### Corrections

Corrections (rename, merge and add speakers; reassign, split, merge and edit
segments; mark passages) run in the engine, each in one transaction that stores
the before/after state of the rows it touches; undo and redo restore those
states in order. The AI output is kept three ways: the `original_transcripts`
copy, `original_text`/`original_speaker_id` per segment and `original_text` per
word. Editing text re-derives the segment's words so word timings stay valid.

Hallucination guards already in Stage 1: VAD, temperature fallback with
compression-ratio / log-probability / no-speech thresholds, prompt reset on
high temperature. Per-segment `avg_logprob`, `no_speech_prob`,
`compression_ratio`, and `temperature` are stored for Stage 3 quality analysis.

Progress reporting never invents numbers: a stage reports a fraction only when it
has a real numerator and denominator; otherwise the UI shows an indeterminate
state with elapsed time. ETA is derived from the measured processing rate and is
shown only after enough work has been observed.

## 6. Processing modes

| Mode | Model | Decoding | Speakers / words |
|------|-------|----------|------------------|
| Fast | Whisper Small | greedy (beam 1) | speakers when the model is installed; Whisper word times |
| Balanced | Whisper Large-v3 Turbo | beam 5 where supported | speakers and forced alignment when installed |
| Maximum Accuracy | Whisper Large-v3 | beam 5 / best-of 5 where supported; float32 on CPU | as Balanced. Multi-pass verification (passes 5–8) arrives in Stage 3; the UI says so. |

Speaker identification can be required, turned off or left automatic per job;
"required" refuses to start without the model instead of silently skipping.

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

Speaker and alignment models (sizes and checksums verified by the same script):

| Model | Download | Est. memory | Languages | Licence |
|-------|---------:|------------:|-----------|---------|
| Speaker diarization: segmentation-3.0 + WeSpeaker ResNet34 | 33 MB | ~0.4 GB | any | MIT; CC BY 4.0 |
| Word alignment: Conformer-CTC Small | 76 MB | ~0.5 GB | English | CC BY 4.0 |
| Word alignment: Conformer-CTC Medium | 166 MB | ~0.8 GB | English | CC BY 4.0 |
| Word alignment: MMS-300M | 1.26 GB | ~2.5 GB | 1,100+ | CC BY-NC 4.0 (non-commercial) |

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
7. **Diarization is a re-implementation.** It uses pyannote 3.1's models and
   algorithm but is not the pyannote.audio library; results are measured against
   reference annotations (DER) rather than assumed equal. Far-field meetings with
   heavy overlap remain hard (see the Stage 2 report). Speaker embeddings
   capture voice, not accent: very similar voices may be merged.
8. **Alignment model licences and coverage.** The most precise aligner (MMS) is
   non-commercial; the permissive ones are English-only. Without an installed
   aligner for the language, word times come from Whisper and the job records
   why. Boundary refinement parameters were fitted on synthetic speech.
9. **YouTube (Stage 4)**: yt-dlp breaks when YouTube changes; current yt-dlp needs
   an external JavaScript runtime for YouTube; terms of service restrict use to
   content the user is authorized to download. No DRM or authentication bypass.
10. **Code signing & notarization (Stage 5)**: every embedded `.so`/`.dylib` must be
    signed with the hardened runtime; numba/llvmlite and MLX may need JIT-related
    entitlements. The Stage 1 bundle is ad-hoc signed and not notarized.
11. **Memory pressure**: the worker unloads Whisper before the diarization and
    alignment models load, and runs them one after the other; their peak memory
    on 16 GB has not been measured yet.
12. **Swift compiler.** Swift 6.1.2 crashed in IR generation on a method
    reference used as a `Binding` setter; closures are used instead. The UI still
    has no automated interaction tests (it is exercised through the model layer,
    the packaged-app self-test and window snapshots).
