# Stage 2 (Speaker Intelligence) — report

All results below come from real runs: real models, real decoding, real audio.
Nothing is mocked. Where something was not verified, it says so. Synthetic
speech (espeak-ng) is used where exact ground truth is needed; it is cleaner
than real recordings, so those numbers describe precision, not real-world
accuracy.

## Implemented

| Requirement | Status |
|-------------|--------|
| Speaker diarization | pyannote 3.1's design and models (segmentation-3.0, WeSpeaker ResNet34) on ONNX Runtime, re-implemented in NumPy/SciPy: sliding 10 s windows, local speakers with overlap, embeddings of clean speech, centroid-linkage clustering with pyannote's tuned threshold, frame-level timeline. No PyTorch, no gated downloads. |
| Estimate the number of speakers | Automatic (clustering threshold); the user may give an exact number or a range. |
| Speaker changes, consistent IDs | Segments are split where the speaker changes (words are never altered; each split is recorded). Speakers are numbered by first appearance and keep one id across the recording. |
| Overlapping speech, interruptions | Overlaps are detected per frame; segments containing them are flagged with the other speakers involved. Short interjections inside a turn are kept when the evidence supports them. |
| Uncertain speaker assignments | Flagged with reasons (low coverage, analysis windows disagree, voice close to another speaker's, no voice under the words, speaker change possibly off by a word). |
| No claimed identities | Speakers are "Speaker 1", "Speaker 2", … until the user names them; nothing sets a name automatically. |
| Forced word alignment | CTC forced alignment (own NumPy Viterbi) with NeMo Conformer-CTC (English) or MMS (1,100+ languages, romanised); wildcard token for numbers/symbols and untranscribed speech; level normalisation for quiet speakers; energy-aware word boundaries. |
| Alignment failures | Rejected alignments keep Whisper's timings and the segment is flagged `alignment_failed` with the reason. |
| Stored per word | text, start, end, speaker, segment, timing source, Whisper's own times, alignment score, original (AI) text. |
| Transcript ↔ video sync | Active segment highlighted and followed during playback, the playing word marked, "who is speaking now" under the video; clicking a segment, timestamp or word seeks the player; "Play Segment" replays only that segment. |
| Speaker renaming and correction | Rename, merge, add speakers; reassign one or more segments (also to a new speaker, which splits a speaker); split a segment before a word; merge adjacent segments; edit text; mark passages uncertain. Every change is one undo step (toolbar, Edit menu ⌘Z / ⇧⌘Z), kept by the engine across app restarts. |
| Original vs corrected | The AI transcript is stored unchanged (`transcript.original`); segments keep `original_text` and `original_speaker_id`, words their `original_text`. |
| Three-panel workspace | Speakers (left) · video (centre) · transcript (right), with a "Needs review" filter for flagged passages. |
| Models | Diarization (33 MB) and alignment models are downloaded only on explicit confirmation, from pinned URLs with SHA-256 verification (or HuggingFace with size and LFS checksum checks); licences are shown before download. |
| Processing view | "Identifying speakers" and "Aligning words" stages with measured progress, or "skipped" with the reason. |
| Export | TXT now groups speaker turns: `[00:01:12] Speaker 1` followed by the text. |

Out of scope (later stages): verification passes, other export formats, full-text search, YouTube.

## Verification

| Where | What ran |
|-------|----------|
| Linux dev container | Engine suite; diarization DER on real recordings; synthetic multi-speaker conversations; word-timing accuracy; full pipeline (Whisper Small + diarization + alignment) with corrections through the RPC methods; review-flag evaluation on the annotated corpus; 13-minute performance run. |
| GitHub Actions, Ubuntu 24.04 | Everything above, plus the MMS multilingual aligner (HuggingFace is not reachable from the dev container) and the word-timing benchmark of every aligner. Models downloaded through the engine's own download path. |
| GitHub Actions, macOS 15 (Apple M1) | Engine suite with faster-whisper, MLX, diarization and alignment; Swift build and tests (fixtures, a real two-speaker transcription driven from the Swift client with rename, reassign, split, undo and redo); app bundle; packaged-app self-tests including a two-person call with speakers, aligned words, a rename and its undo. |

Test counts (CI, final Stage 2 commit):

| Suite | Linux | macOS |
|-------|------:|------:|
| Engine tests without model weights | 78 passed | 77 passed, 1 skipped |
| Engine tests with real models | 24 passed, 3 skipped (MLX needs Apple Silicon) | 25 passed, 2 skipped (MMS is downloaded only on Linux) |
| Swift tests (protocol fixtures, formatting, real engine) | — | 20 passed |
| App bundle: smoke test and self-tests (faster-whisper, MLX, two-person call) | — | 4 passed |

The two-person self-test drives the packaged app end to end: 10 segments,
2 speakers, 80 of 80 words force-aligned, a rename to "Diane", undo back to
"Speaker 1" with redo available, and a snapshot of the three-panel workspace.

## Measured accuracy

### Speaker diarization (DER, lower is better)

DER = (missed speech + false alarm + speaker confusion) / reference speech,
overlap scored, frame-based (10 ms); "collar" excludes ±0.25 s around
reference boundaries.

| Material | Speakers | DER | DER (collar 0.25 s) |
|----------|---------:|----:|--------------------:|
| Real telephone conversation (pyannote sample, 30 s) | 2 | 0.053 | 0.000 |
| 12 AMI meeting excerpts, far-field, 30 s each, heavy overlap | 1–4 | 0.324 | 0.202 |
| Synthetic conversation, 2 voices, overlaps every 4th turn | 2 | — | 0.000 |
| Synthetic conversation, 3 voices, overlaps every 4th turn | 3 | — | 0.006 |
| Synthetic, 3 voices, 12.9 min | 3 | — | 0.003 |

The number of speakers was estimated correctly for the real conversation and
every synthetic one. The AMI excerpts are the hardest material here (distant
microphones, up to four people talking at once, very short turns); most of
their error is missed quiet or overlapped speech. Forcing the true number of
speakers made the AMI result worse (0.43), because the extra speakers have too
little clean speech in 30 s.

### Word attribution (fraction of words given the right speaker)

| Material | Words scored | Correct |
|----------|-------------:|--------:|
| Real conversation, full pipeline (Whisper Small → diarization → alignment) | 68–71 | 98.5–98.6 % |
| Synthetic, 2 voices | 138 | 100 % |
| Synthetic, 3 voices | 138 | 99.3–100 % |

Ranges are over the dev container, Linux CI and macOS CI (Whisper's output
differs slightly between CPU builds, so the word count does too). Words that
fall into overlapping speech in the reference are not scored. In all three
runs every word of the real conversation was force-aligned.

### Do the review flags point at the wrong speaker labels?

`engine/scripts/evaluate_speaker_review.py` runs the full pipeline on every
annotated real recording of the corpus (the 30 s call and 12 AMI excerpts; two
AMI excerpts yield no recognised speech) and checks each word's speaker
against the reference.

| Words | Count | In a segment flagged for review |
|-------|------:|--------------------------------:|
| Right speaker | 442 | 58 (13 %) |
| Wrong speaker | 40 | 1 (2.5 %) |
| Reference ambiguous (overlap or silence) | 115 | 52 (45 %) |

The flags do what they describe: they mark overlapping speech, thin evidence
and uncertain changes, and they concentrate on passages the reference itself
marks as overlapped. They do **not** find the main kind of error measured
here: 37 of the 40 wrong words come from two far-field AMI excerpts in which
both people were found but their voices were confused, with full coverage and
agreement and only a slightly smaller voice margin than correct words. Such
confident confusions need the user's review of the speaker list (play each
speaker's first segment; merge or reassign), not a passage flag. The
"speaker change" flag fired on 12 of the 14 words at speaker changes; none of
those words was wrong, and 8 of them lie in overlapping speech. With so few
errors to learn from, the thresholds were left unchanged.

### Word timestamps (synthetic speech with exact word times)

Benchmark (`engine/scripts/benchmark_word_timing.py`, GitHub Actions Linux):
the input is what the app really has — Whisper Small's own text and word
times — for 12 segments, 137–138 words per voice, three voices (en-us, en-us+f3,
en-gb-scotland). Error is measured against the synthesiser's exact word times.

| Timing source | Mean start error | Starts within 50 ms | Segments aligned |
|---------------|-----------------:|--------------------:|-----------------:|
| Whisper Small alone (cross-attention DTW) | 87–95 ms | 38–47 % | — |
| + alignment, Conformer-CTC Small (default for English) | 36 ms | 73–74 % | 36 / 36 |
| + alignment, Conformer-CTC Medium | 48–63 ms | 55–59 % | 32 / 36 |
| + alignment, MMS-300M | 17–18 ms | 96 % | 36 / 36 |

Whisper's own times run ~75 ms early on average; the aligners' bias is under
±4 ms (Small) or +13–17 ms (MMS). The Medium model is less precise than Small
here (starts ~35 ms late on two voices, and 4 of 12 segments rejected for the
Scottish voice, which then keep Whisper's times); it is last in the automatic
preference order and not recommended. With the reference text instead of
Whisper's (unit test, ±0.15 s jittered input times), MMS measured 20 ms
(English, 73 words), 17 ms (German, 15 words) and 12 ms (Spanish, 14 words).

The boundary rule for connected speech was fitted on two voices and checked on
four others. On real telephone speech the Small aligner initially failed on
most segments; two defects were found and fixed (below), after which every
segment of the 30 s call aligned (76–80 words, depending on the Whisper run).

### Speed and memory (Linux container, 4 vCPU; not the target Mac)

| Step (12.9 min, 3 speakers) | Time | Real-time factor | Peak memory |
|-----------------------------|-----:|-----------------:|------------:|
| Diarization | 78 s | 0.10 | 0.99 GB |
| Alignment (Conformer-CTC Small) | 10.5 s | 0.014 | 0.30 GB |

Speaker embeddings take 88 % of the diarization time. Peak memory on the
target Mac has not been measured.

## Defects found by testing and fixed

1. **Alignment rejected correct text** on real speech: the tokenizer split words
   differently from the model's own SentencePiece segmentation (`▁h el lo`
   instead of `▁he ll o`), so correct words scored like wrong ones. Fixed with a
   frequency-weighted segmentation that reproduces the model's own split for
   98.5 % of words (was 88.5 %).
2. **A quieter speaker was invisible to the aligner** (telephone call, ~7 dB
   quieter): the model normalises features over its input, so the louder voice
   dominated. Fixed with speech-level normalisation of the aligner's input
   (the recording itself is unchanged).
3. **Word boundaries were biased** (starts ~60 ms late, ends ~110 ms early)
   because CTC marks tokens with spikes inside the sounds. Fixed with
   energy-based pause detection and a per-model split of the gap between words
   in connected speech.
4. Word edges were extended over constant noise (no energy contrast). Fixed:
   without contrast, only the connected-speech rule applies.
5. The Swift compiler (6.1.2) crashed on a method reference used as a binding
   setter; replaced by a closure.
6. Database migration: rebuilding the segments table would have cascade-deleted
   words; the migration runs with foreign keys off and checks them afterwards
   (tested on a Stage 1 database).
7. Engine shutdown could close the database while a worker's monitor thread
   was still recording the job's outcome (seen once on macOS CI as a
   background-thread exception). Shutdown now waits for that thread.
8. The packaged-app self-test loaded the transcript without its words, so its
   "≥ 70 % of words aligned" check compared 0 with 0 and passed vacuously. The
   workspace itself always loaded words; the self-test now does too and
   requires them.
9. At the smallest window size (980 pt) the three panels' minimum widths did
   not fit, so the transcript's right edge was clipped and its header wrapped
   letter by letter (seen in the self-test's window snapshot). The smallest
   window is now 1020 pt, the panel minimums fit it (the player keeps the
   300 pt below which its controls drop the timeline), and the transcript
   header puts the model summary on a second line.

## Known limitations

* Far-field meeting audio with heavy overlap remains hard (DER 0.32 on the AMI
  excerpts). Similar or distant voices can be merged into one speaker or
  swapped between two; the fix is the user's merge, reassign and new-speaker
  tools.
* The alignment models are English-only (permissive licences) or multilingual
  but **non-commercial** (MMS, CC BY-NC 4.0); without a suitable model, word
  times come from Whisper and the job records why. Boundary parameters were
  fitted on synthetic speech.
* Uncertainty flags are heuristics, not calibrated probabilities. On the real
  recordings they flagged 1 of 40 wrong-speaker words: confident confusions of
  two voices (far-field audio) are not flagged, and speaker changes in
  conversation are flagged even when right.
* Text edits re-time new words by interpolation, not by re-alignment.
* Peak memory and speed on the target M2 Pro are not measured; diarization runs
  on the CPU (0.10× real time on 4 vCPU here).
* The UI has no automated interaction tests; it is exercised through its model
  layer, the packaged-app self-test (which drives a rename and undo and
  snapshots the three-panel workspace) and the Swift integration tests.
* Starting a new transcription replaces the transcript and discards
  corrections (the app warns before re-transcribing an edited transcript).

## Launching

```sh
scripts/build_app.sh && open build/Verbatim.app
```

Then **Settings › Models**: download a Whisper model, **Speaker
identification** (33 MB) and an alignment model (English: Conformer-CTC Small,
76 MB; other languages: MMS, 1.26 GB, non-commercial). Drop a file in the
Library, keep **Identify speakers** on (optionally set the number of speakers)
and press **Transcribe**.

## Recommended next steps (Stage 3)

1. Quality analysis on top of the Stage 2 evidence: combine Whisper confidence,
   alignment scores and speaker uncertainty into review priorities, and add a
   recording-level warning when speakers are hard to tell apart (the confident
   voice confusions above), calibrated on more annotated recordings with
   `evaluate_speaker_review.py`.
2. Targeted re-transcription of flagged passages with padding, and re-alignment
   of edited text.
3. Measure memory and speed on an M2 Pro; consider CoreML/MPS for the speaker
   embeddings, which dominate diarization time.
4. A larger real benchmark with reference transcripts and speaker annotations
   (podcasts, interviews) under compatible licences.
