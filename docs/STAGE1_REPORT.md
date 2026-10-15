# Stage 1 (Foundation) — report

All results below come from real runs: real Whisper weights, real decoding, real
audio. Nothing is mocked. Where something was not verified, it says so.

## Implemented

| Requirement | Status |
|-------------|--------|
| Native SwiftUI app (Swift 6, macOS 13+) | Sidebar (Library, Recent Projects, Processing, Completed, Settings), import panel, processing view, transcript workspace, Settings (models, engine). Builds with zero compiler warnings. |
| Media import | Drag and drop or file picker; every file is validated and probed by the engine before a project is created (regular file, readable, decodable, has an audio stream; cover art is not treated as video). |
| FFmpeg integration | In-process through PyAV (FFmpeg 7 libraries): no shell, no `ffmpeg` on `PATH`. Streaming decode to 16 kHz mono WAV that keeps the media timeline (offsets, gaps, discontinuities). |
| Real Whisper transcription | faster-whisper (CPU, beam search) and mlx-whisper (Apple GPU, greedy), behind a capability-aware interface. VAD, temperature fallback, word timestamps, seeded RNGs for reproducibility; the settings that actually ran are recorded per job. |
| Long recordings | Chunked at pauses (each chunk's audio stops at the pause, so nothing is heard twice); alignment-based splicing only where a cut must fall inside speech; previous-chunk text as context; first-chunk lead-in. |
| Timestamps and transcript display | Timestamped segments, detected language, which model/backend produced the transcript; clicking a timestamp seeks the player. |
| TXT export | Header (title, source, duration, language, model) and `[HH:MM:SS]` lines; atomic write via the save panel. |
| Persistent project storage | SQLite (WAL, migrations): projects, jobs, stages, chunks, segments (original text kept separately), words, settings. |
| Jobs | One at a time (memory), worker subprocess per job, cancel, resume from completed chunks, recovery after the app quits or crashes. Progress is shown as a fraction only when measured; ETA only after enough work is observed. |
| Model management (basic) | Catalog of Small/Medium/Large-v3/Large-v3-Turbo for both backends; explicit download with size shown and confirmation; delete; models folder selectable (external SSD). Nothing downloads implicitly. |
| Packaging | `scripts/build_app.sh` builds `Verbatim.app` with an embedded relocatable CPython and the locked engine; launches from Finder with no Terminal setup. |

Processing modes exist (Fast / Balanced / Maximum Accuracy), but in Stage 1 they
only choose the model and decoding settings. The UI says that the multi-pass
verification is not implemented yet.

## Verification

| Where | What ran |
|-------|----------|
| Linux dev container | Engine test suite with real Whisper Small (int8) weights. |
| GitHub Actions, Ubuntu 24.04 | Engine suite; official Whisper Small downloaded through the engine's own download path; HuggingFace catalog verification. |
| GitHub Actions, macOS 15.7 on Apple M1 (Xcode 16.4, Swift 6.1.2, Metal available) | Engine suite with faster-whisper **and MLX**; `swift build`; 17 Swift tests including a real transcription driven from the Swift client and engine-crash handling; `Verbatim.app` build (795 MB); Finder-style launch smoke test (app starts its engine, quits cleanly, engine exits); full **packaged-app self-test** with both backends: import a video → transcribe → open the workspace → snapshot → quit. |

Test counts: 65 engine tests (50 without weights, 15 with real weights, 3 of
those MLX-only), 17 Swift tests, 2 packaged-app self-tests.

## Measured accuracy (Whisper Small, word error rate)

WER uses the documented normaliser in `engine/src/verbatim_engine/metrics/wer.py`
(case and punctuation removed, numbers not normalised). Only the JFK sample is
real human speech; the other fixtures are synthetic speech (espeak-ng on Linux,
macOS `say`) and are **not representative of real-world accuracy**. They exist to
test the pipeline (chunking, boundaries, resume) with known text.

| Fixture | Linux, faster-whisper int8 | macOS, faster-whisper int8 | macOS, MLX fp16 |
|---------|---------------------------:|---------------------------:|----------------:|
| JFK inaugural excerpt (real speech, 22 words) | 0.000 | 0.000 | 0.000 |
| JFK as video through the full pipeline | 0.000 | 0.000 | 0.000 |
| 12 sentences, chunked at pauses (143 words) | 0.007 | 0.077 | — |
| Abrupt onset, speech from sample 0 (37 words) | 0.000 | 0.000 | — |
| Continuous speech, single pass (135 words) | 0.030 | 0.059 | — |
| Same, forced cuts inside speech | 0.037 | 0.081 | — |
| Cancel, then resume (143 words) | 0.000 | 0.084 | — |
| App quit mid-job, then resume (75 words) | 0.000 | 0.040 | — |
| German sentence (language detected: de) | 0.154 | 0.077 | — |
| Spanish sentence (language detected: es) | 0.077 | 0.231 | — |

The macOS synthetic-speech errors are recognition errors on the `say` voice
("after a long" → "on", "comment during an" → "commentary and"). They also occur
in single-pass runs, and every test of boundary handling passed on both
platforms: merged words equal the chunks' words at pause cuts, and there are no
lost or duplicated words at cuts inside speech. Diarization error rate (DER) and
word-timestamp accuracy are Stage 2 metrics and were not measured.

## Defects found by testing and fixed

1. Words dropped at chunk boundaries: Whisper stretches the first word of a
   window back to the window start, and the original midpoint/time rules
   discarded it. Fixed by giving chunks pause-bounded audio, plus alignment-based
   splicing for cuts inside speech.
2. A first word at sample 0 could be dropped (macOS voice). Fixed with a silent
   lead-in on the first chunk.
3. App quit was recorded as "cancelled" instead of "interrupted". Fixed with
   separate signals and recorded stop reasons.
4. Results varied between runs (unseeded temperature-fallback sampling). Fixed by
   seeding.
5. **The packaged app crashed when opening a project** (SwiftUI `VideoPlayer`
   without AVKit linked). Fixed with an `AVPlayerView` representable and explicit
   linking.
6. **The app could deadlock while quitting** when termination was requested from
   main-actor work. Fixed by stopping the engine off the main actor.

## Known limitations

* No speaker labels, word alignment, verification passes, YouTube import, editing,
  SRT/VTT/JSON/CSV/Markdown export or full-text search yet (Stages 2–4).
* The UI has no automated interaction tests (drag and drop, buttons). It was
  verified by compiling, the self-test (which drives the app's model layer and
  renders the workspace) and window snapshots; a person should still click
  through it once.
* Peak memory per model is estimated, not measured. Large-v3 on the CPU backend
  (float32 in Maximum Accuracy mode) is slow; the MLX backend has no beam search.
* MLX needs macOS 14+ with current wheels; macOS 13 gets the CPU backend only.
* The bundle is ad-hoc signed and not notarized, and is about 800 MB. The PyAV
  FFmpeg build includes GPL components (x264/x265) according to its configuration;
  the licence position must be resolved before distribution
  (`Contents/Resources/BUILD_INFO.txt` records the build).
* Audio-only MP4 files whose first sample is not at time 0 are aligned to their
  first audio sample.
* The app is not sandboxed (it launches its engine and reads user-chosen files).

## Launching

```sh
scripts/build_app.sh && open build/Verbatim.app
```

Then **Settings › Models › Download…** (for example Whisper Large-v3 Turbo for
MLX), drop a file in the Library, pick a mode and press **Transcribe**. The CI
artifact `Verbatim-app-unsigned` contains the same bundle; it is quarantined
when downloaded, so open it with right-click › Open.

## Recommended next steps (Stage 2)

1. Speaker diarization with pyannote.audio (gated models: one-time HuggingFace
   token and licence acceptance), with memory measured on 16 GB and run
   sequentially after ASR.
2. CTC forced alignment for word timestamps (self-contained Viterbi; `torchaudio`'s
   aligner is deprecated); keep Whisper DTW timings as a fallback and flag failures.
3. Synchronised highlighting (the transcript index already exists), speaker
   panel, editing with undo, and a corrections table alongside `original_text`.
4. A real multi-speaker corpus with reference transcripts and RTTM for DER
   (e.g. AMI / VoxConverse excerpts under compatible licences).
5. Pin model revisions in the catalog (CI prints the commit hashes) and measure
   peak memory and real-time factor for each model on an M2 Pro.
