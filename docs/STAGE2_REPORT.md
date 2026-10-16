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
| Linux dev container | Engine suite; diarization DER on real recordings; synthetic multi-speaker conversations; word-timing accuracy; full pipeline (Whisper Small + diarization + alignment) with corrections through the RPC methods; 13-minute performance run. |
| GitHub Actions, Ubuntu 24.04 | Everything above, plus the MMS multilingual aligner (HuggingFace is not reachable from the dev container) and the word-timing benchmark of every aligner. Models downloaded through the engine's own download path. |
| GitHub Actions, macOS 15 (Apple M1) | Engine suite with faster-whisper, MLX, diarization and alignment; Swift build and tests (fixtures, a real two-speaker transcription driven from the Swift client with rename, reassign, split, undo and redo); app bundle; packaged-app self-tests including a two-person call with speakers, aligned words, a rename and its undo. |

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
| Real conversation, full pipeline (Whisper Small → diarization → alignment) | 68 | 98.5 % |
| Synthetic, 2 voices | 138 | 100 % |
| Synthetic, 3 voices | 138 | 99.3 % |

Words that fall into overlapping speech in the reference are not scored.

### Word timestamps (synthetic speech with exact word times)

Mean absolute error of word start times; 138 words per voice, three voices.

| Timing source | Mean start error | Starts within 50 ms |
|---------------|-----------------:|--------------------:|
| Whisper Small alone (cross-attention DTW) | ~90 ms | ~45 % |
| + alignment, Conformer-CTC Small | 34 ms | 73–77 % |
| + alignment, Conformer-CTC Medium | ~44–59 ms | ~55–67 % |
| + alignment, MMS-300M (English) | 18 ms | 96 % |
| MMS-300M, German / Spanish | 17 ms / 12 ms | 93 % / 100 % |

The boundary rule for connected speech was fitted on two voices and checked on
four others. On real telephone speech the Small aligner initially failed on
most segments; two defects were found and fixed (below), after which 11 of 12
segments aligned.

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

## Known limitations

* Far-field meeting audio with heavy overlap remains hard (DER 0.32 on the AMI
  excerpts). Very similar voices can be merged into one speaker; the fix is the
  user's merge/split tools.
* The alignment models are English-only (permissive licences) or multilingual
  but **non-commercial** (MMS, CC BY-NC 4.0); without a suitable model, word
  times come from Whisper and the job records why. Boundary parameters were
  fitted on synthetic speech.
* Uncertainty thresholds are heuristics, not calibrated probabilities.
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
   alignment scores and speaker uncertainty into review priorities.
2. Targeted re-transcription of flagged passages with padding, and re-alignment
   of edited text.
3. Measure memory and speed on an M2 Pro; consider CoreML/MPS for the speaker
   embeddings, which dominate diarization time.
4. A larger real benchmark with reference transcripts and speaker annotations
   (podcasts, interviews) under compatible licences.
