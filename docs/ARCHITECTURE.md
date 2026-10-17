# Verbatim — Architecture

Status: **Stage 4 (Media Import, Search, Exports, Queue)** on top of Stages
1–3 (Foundation, Speaker Intelligence, Accuracy Verification). This document
records the selected libraries, their verified compatibility, model
requirements, and known risks. Anything not yet verified is marked as such.
Measured results are in `docs/STAGE1_REPORT.md` … `docs/STAGE4_REPORT.md`.

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
                      ├─ Export Manager            TXT, SRT, WebVTT, JSON, CSV, Markdown
                      ├─ Search                    SQLite FTS5 index kept in step by triggers
                      └─ Job Manager (persistent queue: order, pause, auto-resume)
                           └─spawns──▶ verbatim-engine worker --job <id>   (one at a time)
                                         [download (YouTube)] → extract → plan → load model →
                                         transcribe chunks → (unload Whisper) → diarize → align →
                                         quality → re-check → validate → finalize
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
| 1 | Media Import Manager | 1 / 4 | `engine/src/verbatim_engine/media/` (probe + validation), `youtube/` (URL validation, yt-dlp retrieval, captions), `app/Sources/Verbatim/Views/LibraryView.swift`, `YouTubeImport.swift` |
| 2 | Audio Processing Engine | 1 | `engine/src/verbatim_engine/audio/` (decode → 16 kHz mono WAV, chunk planner, WAV reader, level normalisation) |
| 3 | Speech Recognition Engine | 1 | `engine/src/verbatim_engine/asr/` (backend protocol, faster-whisper, mlx-whisper, merge) |
| 4 | Speaker Diarization Engine | 2 | `engine/src/verbatim_engine/diarization/` (segmentation, embeddings, clustering, timeline, word/segment assignment) |
| 5 | Word Alignment Engine | 2 | `engine/src/verbatim_engine/alignment/` (CTC models, Viterbi, aligner, boundary refinement) |
| 6 | Transcript Verification Engine | 3 | `engine/src/verbatim_engine/verify/` (quality analysis, re-transcription, acoustic comparison, validation, review priority), `alignment/score.py` (CTC forward scoring), `llm/` (optional AI review: providers, prompts, validation, runs) |
| 7 | Transcript Editor | 2 / 4 | `engine/src/verbatim_engine/transcript/` (data model, build, corrections + undo), `search.py` (full-text search), `app/Sources/Verbatim/Workspace/`, `Views/SearchView.swift` |
| 8 | Export Manager | 1 (TXT) / 4 | `engine/src/verbatim_engine/export/` (TXT, SRT, VTT, JSON, CSV, Markdown) |
| 9 | Model Manager | 1–2 / 5 | `engine/src/verbatim_engine/models/` (HuggingFace and pinned-URL downloads) |
| 10 | Persistent Job Manager | 1 / 4 | `engine/src/verbatim_engine/jobs/` (queue order, pause, auto-resume), `Views/ProcessingView.swift` |

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
| Storage | SQLite (Python `sqlite3`) | system | WAL, foreign keys, migrations (schema 6); FTS5 full-text index (`unicode61 remove_diacritics 2`) | Linux + CI |
| Diarization | pyannote segmentation-3.0 + WeSpeaker ResNet34 (the models of pyannote's `speaker-diarization-3.1`), as ONNX exports, run on onnxruntime; pipeline re-implemented in NumPy/SciPy | pinned SHA-256 | Same models and algorithm as pyannote 3.1 without PyTorch (excluded from the bundle) and without gated HuggingFace downloads | Linux + CI (DER measured) |
| Clustering | SciPy (`linkage`, `linear_sum_assignment`) | `scipy>=1.13` | centroid-linkage agglomerative clustering; already in the bundle via mlx-whisper | Linux + CI |
| Speaker features | kaldi-native-fbank | `>=1.20` | Kaldi-compatible filterbanks (WeSpeaker and NeMo inputs), Apache-2.0 | Linux + CI |
| Alignment (English) | NVIDIA NeMo Conformer-CTC (Small, Medium), ONNX exports by sherpa-onnx | pinned SHA-256 | CC BY 4.0; testable offline; ~35 ms word-start error | Linux + CI |
| Alignment (multilingual) | Meta MMS-300M forced aligner (ONNX) | HF file, size-verified | 1,100+ languages; most precise measured (~18 ms); **CC BY-NC 4.0** | CI only (HF unreachable from the dev container) |
| Romanisation | uroman | `>=1.3` | non-Latin scripts → Latin letters for MMS (Apache-2.0) | Linux |
| CTC alignment | own NumPy Viterbi | — | `torchaudio.functional.forced_align` is deprecated and needs PyTorch | Linux + CI |
| Video retrieval | **yt-dlp** (Python API, in-process) | `yt-dlp==2026.8.19` | authorized content only; metadata, formats, captions; no cookies, sign-in, geo bypass or DRM formats | Linux (real yt-dlp against a local page); YouTube itself: CI attempt, see the Stage 4 report |
| YouTube player challenges | **yt-dlp-ejs** + **deno** | `yt-dlp-ejs==0.8.0`, `deno==2.9.7` (PyPI wheel with the deno binary) | YouTube's formats need its player JavaScript solved; yt-dlp runs the pinned EJS scripts in deno (`remote_components` empty: no scripts fetched at run time) | Linux (deno runs), macOS bundle check |

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
9. **Speakers to words** — each word gets the speaker active for most of its
   duration; segments are split where the speaker changes; doubtful
   attributions are flagged.
10. **Quality analysis (pass 5)** — a speech map of the recording on a 20 ms
    grid (Silero VAD at fine settings, plus the diarization's speech activity)
    and per-segment detectors flag unclear speech, possible hallucinations,
    repetitions and untranscribed speech (rules below). Each flag has a
    severity; a segment's review priority is its most severe open flag.
11. **Targeted re-transcription (pass 6, Maximum Accuracy)** — passages with a
    medium or high doubt are decoded again from the original 16 kHz recording
    with 2 s of context on each side (windows merged up to 28 s): with the
    preceding 48 words as prompt and beam ≥ 5 at temperature 0 ("context"),
    without prompt ("plain"), and, for possible hallucinations, with VAD and
    Whisper's hallucination-silence threshold ("vad"). Passages that overlap
    another speaker's segment by ≥ 0.25 s are not re-worded (recorded as not
    re-checked): acoustic evidence cannot separate simultaneous voices. Decodes
    are appended to `recheck.jsonl` (resume).
12. **Acoustic comparison (pass 7)** — every candidate wording (first pass and
    each decode, cut to the passage by aligning the decode with the
    transcript's own words) is scored as log P(words | audio) with the CTC
    forward algorithm on the word-alignment model's emissions, between the same
    anchor words, alongside "nothing was said" (rules below). Changed segments
    are re-aligned, given speakers again and re-analysed.
13. **Validation (pass 8)** — timestamps clamped to the recording and to their
    words, word order checked, boundary duplicates flagged, untranscribed speech
    recomputed on the final transcript, review counts summarised. Nothing is
    deleted or re-worded here; unresolved doubts stay on the segments.
14. **Finalize** — speakers are numbered by first appearance; the transcript,
    its immutable original copy and the speakers are written in one transaction.

### Quality rules (pass 5; heuristics, thresholds in `verify/quality.py`)

| Flag | Rule | Severity |
|------|------|----------|
| `low_confidence` | average log-probability < −1.0; or ≥ 30 % (and ≥ 2) of the words with probability < 0.5; or the decoder fell back to sampling | high below −1.3 or with half the words uncertain, else medium |
| `possible_hallucination` | < 20 % of the word time over detected speech; alignment rejected because the text does not match; a repetition loop or compression ratio > 2.4; no-speech probability > 0.6; a phrase Whisper is known to produce on silence | high for the first three, low for a known phrase alone, else medium |
| `repeated_words` | an immediately repeated 1–4-word sequence (often genuine) | low |
| `missing_speech` | ≥ 1 s of VAD speech no word covers (0.25 s margin around words) | high with both detectors or ≥ 3 s, else medium |

### Acoustic comparison rules (pass 7; `verify/compare.py`)

* The first-pass wording is kept unless another candidate scores at least
  3 nats higher, **plus 3 nats for every word the candidate has fewer**: CTC
  scoring favours dropping words the model is unsure of (measured on AMI).
* Wordings within the margin are kept as **alternatives** for the user;
  agreement between decodes is never taken as evidence.
* If "nothing was said" explains the audio better than every wording by the
  margin, the passage is flagged as a possible hallucination (**words are never
  removed**). Over detected speech (≥ 50 % of the words) this is a medium doubt,
  since the acoustic model may not recognise overlapping, distant or telephone
  speech; elsewhere it is high.
* Untranscribed speech gets words only if they beat "nothing" by the margin.
* User-corrected passages are never re-worded; a better-supported wording is
  offered as an alternative. Every outcome is recorded (`rechecked` flag with
  the scores of all candidates).
* Without an acoustic model for the language, differing wordings are listed for
  review and nothing changes.

The margin comes from measurements on real speech: single-word substitutions,
insertions and repetitions of the true text always scored lower than the truth
(by ≥ 3.5 nats on JFK and LibriSpeech); deleting a short word scored up to
1.2 nats better.

### Optional AI review (`llm/`)

A language model may flag suspicious passages, suggest punctuation, point out
inconsistent spellings of a term and suggest passages to re-transcribe. It never
changes the transcript: suggestions are validated (a punctuation suggestion may
not change words; a terminology suggestion must name a spelling that occurs in
the passage), stored with the run's every request and response, and applied
only when the user accepts one — as an undoable edit. "Re-transcribe" starts an
acoustic re-check, so the audio decides, not the language model. Providers sit
behind one interface: Ollama (local, `/api/chat` with a JSON schema) and OpenAI
(Chat Completions with structured output). Local endpoints are the default; a
remote endpoint needs HTTPS and an explicit permission per review.

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

Stage 3 corrections: choose a recorded wording, mark passages reviewed, remove
a passage, and re-check chosen passages on demand (a job of kind `recheck`
using the audio extracted by the transcription; its outcome is one undoable
correction).

Hallucination guards from Stage 1 remain: VAD, temperature fallback with
compression-ratio / log-probability / no-speech thresholds, prompt reset on
high temperature.

Progress reporting never invents numbers: a stage reports a fraction only when it
has a real numerator and denominator; otherwise the UI shows an indeterminate
state with elapsed time. ETA is derived from the measured processing rate and is
shown only after enough work has been observed.

### Stage 4: video import, search, exports and the queue

* **Video import** (`youtube/`). `urls.validate` accepts one YouTube video and
  turns it into the canonical watch URL before any network access.
  `youtube.inspect` reads the metadata (title, channel, licence, duration,
  chapters, caption tracks, sizes) and says when a video cannot be imported
  (live, needs an account, DRM). The user must confirm they are authorized
  (`projects.createFromYouTube` refuses otherwise; the confirmation and its
  wording are stored with the project). The `youtube` job's first stage
  retrieves AAC audio and, if wanted, H.264 video up to 720p, joins them
  without re-encoding (PyAV remux), and continues partial downloads after an
  interruption; then the normal pipeline runs on the file. A local file is
  always the fallback: anything that cannot be retrieved can be imported.
* **Captions** are a separate source: stored as published (`captions` table),
  exportable as SRT/VTT, and compared with the transcript after it is made.
  Where they differ clearly (word-sequence agreement < 0.6 over passages of
  at least 3 words) the passage gets a low `captions_differ` flag that shows
  the caption wording. Captions never replace or fill in transcript words:
  automatic captions come from another recogniser, creators' captions are
  often condensed.
* **Advertisements.** Ads YouTube inserts during playback are not in the
  retrieved media and are not transcribed (the app says so before import).
  Sponsor messages inside the video are speech in the media and are
  transcribed; passages inside chapters the creator titled as sponsor or ad
  segments get an informational `in_sponsor_chapter` flag. Verbatim does not
  detect unmarked sponsor messages and does not claim to.
* **Search** (`search.py`, schema 4): FTS5 external-content table over
  segment text, kept in step by triggers; the user's text is tokenised and
  quoted by the engine (never FTS5 syntax), words match as prefixes;
  languages without spaces fall back to substring search. The app searches
  the library or the open transcript (⌘F), highlights matches and jumps to
  the passage.
* **Exports** (`export/`): subtitle cues are cut from word times; the JSON
  export carries metadata, speakers, segments, words, flags and the
  original AI transcript; CSV is defused against formula injection.
* **Persistent queue** (schema 5): `queue_position` orders queued jobs; the
  paused state and auto-resume setting are stored, so the queue survives a
  restart. Pausing can let the current job finish or interrupt it (it keeps
  its progress and goes back to the front). Interrupted jobs are queued again
  at startup when auto-resume is on (at most 5 automatic attempts per job).

## 6. Processing modes

| Mode | Model | Decoding | Speakers / words |
|------|-------|----------|------------------|
| Fast | Whisper Small | greedy (beam 1) | speakers when the model is installed; Whisper word times |
| Balanced | Whisper Large-v3 Turbo | beam 5 where supported | speakers and forced alignment when installed |
| Maximum Accuracy | Whisper Large-v3 | beam 5 / best-of 5 where supported; float32 on CPU | as Balanced, plus re-transcription of doubtful passages compared by acoustic evidence (passes 6–7) |

Quality analysis (pass 5) and validation (pass 8) run in every mode, so every
transcript shows its doubts. Re-checking can also be forced on or off per job
(`recheck`).

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
* Video import connects to YouTube only when the user asks, after they
  confirm they are authorized to retrieve the video. yt-dlp runs with no
  cookies, no account, no `.netrc`, no geo bypass, no unplayable or DRM
  formats, only the YouTube extractor, and no scripts fetched at run time;
  URLs are validated and canonicalised first. Retrieved media lives in
  `<data dir>/media/<project id>/` and is removed with the project.
* AI review is optional and off until configured. Nothing is sent to a cloud
  service without explicit authorization: the OpenAI provider is disabled until
  the user allows it in Settings, and every review of a project with a remote
  endpoint asks again before any text leaves the Mac. Remote endpoints must use
  HTTPS; local (loopback) endpoints are the default. The OpenAI API key is kept
  in the macOS Keychain and passed to the engine only for the request that
  needs it; it is never stored by the engine or written to logs. Audio is never
  sent: only transcript text, in batches the run records.

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
9. **YouTube**: yt-dlp breaks when YouTube changes, and pinned versions age
   quickly (updating yt-dlp is a Stage 5 maintenance concern). YouTube often
   asks data-centre and some home addresses to sign in ("not a bot"); Verbatim
   never signs in, so such requests fail with `youtube_blocked` and the user
   imports a file instead. Retrieval from YouTube itself could not be tested
   from the development container (no route to YouTube); CI attempts it
   without blocking the build. Use is limited to content the user is
   authorized to download; the confirmation is the user's, Verbatim cannot
   verify it. The bundled deno adds about 90–100 MB, and its JIT will need the
   matching hardened-runtime entitlement for notarization (Stage 5).
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
13. **Acoustic evidence has limits.** The English acoustic model (Conformer-CTC
    Small) does not recognise espeak's synthetic voice and is weak on overlapping,
    far-field and narrow-band speech; it cannot tell a foreground speaker from
    background speech. Re-checking therefore never removes words, skips
    overlapping passages and lowers its doubt over detected speech. Outside
    English, acoustic comparison needs the MMS model (non-commercial licence).
14. **Review flags are heuristics.** Their recall and precision are measured by
    the accuracy benchmark (Stage 3 report), not guaranteed; a passage without a
    flag is not certified correct.
15. **AI review was tested with a small local model only** (CI pulls a 0.5 B
    model into Ollama); the OpenAI provider is tested against Ollama's
    OpenAI-compatible endpoint, not against OpenAI's service.
