# Stage 3 (Verification and accuracy) — report

All results below come from real runs: real models, real decoding, real audio.
Nothing is mocked. Where something was not verified, it says so. Synthetic
speech (espeak-ng) is cleaner than real recordings; results on it describe
behaviour, not real-world accuracy. No accuracy figure here is estimated:
each one is measured, and the command that measures it is given.

## Implemented

| Requirement | Status |
|-------------|--------|
| Processing modes | Fast, Balanced, Maximum Accuracy. Quality analysis (pass 5) and validation (pass 8) run in every mode; Maximum Accuracy adds targeted re-transcription with acoustic comparison (passes 6–7). Re-checking can be forced on or off per job; the start sheet says whether it will run and whether wordings can be compared against the audio. |
| Pass 5 — quality analysis | A speech map of the whole recording (Silero VAD on a 20 ms grid plus the diarization's speech activity) and per-segment detectors: unclear speech (low average log-probability, many uncertain words, decoder fallback), possible hallucinations (words over no detected speech, text the aligner cannot find in the audio, repetition loops / high compression, high no-speech probability, phrases Whisper produces on silence), repeated words, untranscribed speech. Overlapping speakers, uncertain speaker assignments and alignment failures come from Stage 2 and feed the same review priority. |
| Pass 6 — targeted re-transcription | Doubtful passages are decoded again from the original 16 kHz recording with 2 s of context on each side, in up to three configurations (with the preceding words as prompt and beam ≥ 5; without prompt; with VAD and the hallucination-silence threshold for possible hallucinations). Decodes are cached per job (an interrupted job resumes). Passages overlapping another speaker are not re-worded. |
| Pass 7 — candidate comparison | Each wording is scored as log P(words \| audio) by the CTC forward algorithm on an independent acoustic model (the word-alignment model), between shared anchor words, alongside "nothing was said". A wording replaces the first pass only with a 3-nat lead plus 3 nats per dropped word; close wordings are kept as alternatives; words are never removed automatically; agreement between decodes is not evidence. Numbers are scored by their spoken forms (years, currency, percentages, ordinals). |
| Pass 8 — validation | Timestamps clamped to the recording and the words, word order, boundary duplicates, untranscribed speech recomputed on the final transcript (also when the transcript is empty), review summary. Unresolved doubts stay on the segments. |
| Targeted re-transcription on demand | Re-check any passages from the transcript (context menu or Review menu): a job of kind `recheck` with the transcription's settings and extracted audio; user-corrected passages are never re-worded; the outcome is one undoable correction. |
| Uncertainty indicators | Every segment carries a review priority (high / medium / low) from its most severe open flag; the Review menu filters the passages that need review, most severe first; words below 50 % probability are coloured; flag badges explain each doubt and every re-check outcome with the candidates and their scores; "Use Wording", "Mark Reviewed", "Remove Passage" (undoable). An empty transcript over detected speech says where speech was not recognised instead of "no speech". |
| Optional LLM verification | Ollama (local) and OpenAI behind one provider interface. The model may flag suspicious passages, suggest punctuation, point out inconsistent spellings and suggest passages to re-check. Suggestions are validated (punctuation may not change words; a spelling must occur in the passage), stored with every request and response, and applied only when accepted, as an undoable edit; "re-check" suggestions start an acoustic re-check. Local by default; a remote endpoint needs HTTPS, the user's permission in Settings and a confirmation per review. The OpenAI key is kept in the macOS Keychain. |
| Accuracy benchmarks | `engine/scripts/benchmark_accuracy.py` runs the real pipeline (engine service, worker processes, all passes) on a reproducible corpus (`engine/tests/support/corpus.py`: pinned, checksummed downloads and deterministic synthesis) and measures WER, CER, DER, word-start error, words produced on silence, how review flags relate to wrong words, and interrupted processing; once with one pass and once with re-checking. CI runs it on every push. |
| Protocol | Version 3 (`docs/ENGINE_PROTOCOL.md`); fixtures captured from a real engine run are decoded by the Swift tests. |

Out of scope (later stages): YouTube import, caption retrieval, search, other
export formats, persistent processing queues.

## Verification

| Suite | Result |
|-------|--------|
| Engine unit, storage and RPC tests (no models) | 144 passed (Linux); 143 passed, 1 skipped (macOS) |
| Engine real-model tests | Linux: 30 passed, 4 skipped (3 MLX tests need Apple Silicon; in the AI-review test the model made no applicable suggestion, so the accept step was skipped). macOS: 30 passed, 4 skipped (MMS model and Ollama not installed there) |
| Swift tests (protocol fixtures, real engine, real transcription and corrections) | 22, 0 failures |
| Packaged-app self-tests on macOS | faster-whisper, MLX, and Maximum Accuracy with speakers and alignment on the two-person call: 80 / 80 words aligned, speaker renamed and undone, two passages re-checked on demand through the app's client ("Re-check 2 passages: First-pass wording kept everywhere.") |
| Accuracy benchmark | runs in CI on every push (Whisper Small); also run locally with Tiny and Small (int8) |

Final Stage 3 CI run, all green on Linux and Apple Silicon macOS:
https://github.com/sematrica/otis/actions/runs/39000420566

New Stage 3 tests include: the CTC forward algorithm against brute-force path
enumeration; every decision rule of the comparison (margin, deletion cost,
empty decodes, unspellable words); the quality detectors; review priority;
re-checking a passage twice; an empty first pass over real speech; the
Maximum Accuracy pipeline end to end followed by an on-demand re-check that
keeps a user's correction; choosing wordings, review marks and removal with
undo; AI suggestions (validation, outdated suggestions never applied, a
decision cannot be overwritten, a remote endpoint refused without permission);
and an AI review against a real local model (Ollama, `qwen2.5:0.5b`).

## Measured accuracy

`engine/scripts/benchmark_accuracy.py`, all 28 corpus items, each run twice
with the same model and decoding settings: **one pass** (quality checks and
validation) and **re-checked** (plus passes 6–7). WER is over all reference
words of a category.

### Word error rate by category

| Category (words) | Small, CI | Small int8, local | Tiny, local |
|---|---|---|---|
| Single speaker, clean (99) | 3.03 % → 3.03 % | 3.03 % → 3.03 % | 5.05 % → 5.05 % |
| Background noise (88) | 57.95 % → 56.82 % | 50.00 % → 46.59 % | 80.68 % → 77.27 % |
| Overlapping speech (194) | 25.77 % → 25.26 % | 22.16 % → 20.62 % | 32.99 % → 31.44 % |
| Long silences (22) | 0 % → 0 % | 0 % → 0 % | 0 % → 0 % |
| Two speakers, synthetic (75) | 0 % → 0 % | 0 % → 0 % | 1.33 % → 1.33 % |
| Three speakers, synthetic (75) | 1.33 % → 1.33 % | 1.33 % → 1.33 % | 2.67 % → 2.67 % |
| Multilingual, synthetic de/es/fr (65) | 33.85 % → 33.85 % | 33.85 % → 33.85 % | 70.77 % → 70.77 % |
| 30 s of faint noise | 0 words → 0 words | 0 → 0 | 0 → 0 |

What this shows:

* **Re-checking helps only on hard audio, and only a little.** All the gains
  come from the babble items (JFK over three other real speakers), where
  wrongly inserted words were removed. Clean speech, silences, conversations
  and the multilingual items are unchanged because nothing there was doubtful
  enough to re-check, or no other wording fitted the audio clearly better.
* **It never made WER worse** in these runs: no correct word was replaced.
* What re-checking changed (Tiny): on the call, "This is daddy and a new
  jersey" became "this is Diane and New Jersey" (Whisper Small hears "Diane
  in New Jersey"; the acoustic model's lead was 42 nats). On the babble item,
  inserted "these are" and "lot of" were removed. On AMI excerpts without a
  verified transcript, "video with symbol." became "idea of what's involved."
  and "Adam." became "us." — correctness unknown.

### Speaker error (DER, collar 0.25 s)

Unchanged by re-checking in both Small runs on every item (call 0.015, AMI
excerpts 0.08–0.71, synthetic conversations 0.000). With Tiny, two AMI
excerpts changed: trn05 0.0705 → 0.0656 (better) and trn08 0.549 → 0.595
(worse). In trn08 the only difference between the two transcripts is one
re-aligned word ("Adam." → "us."); DER is computed from the words' time
spans, so that word's new span and speaker account for the change. Two AMI
excerpts (trn01, trn02) have DER 1.0: their annotation has only short turns
(0.4–1.5 s; trn02 a single 0.7 s turn at −65 dBFS), VAD detects no speech in
either, and no words are produced. The benchmark now counts an empty transcript as missing
all reference speech instead of leaving DER out.

### Do the uncertainty indicators point at the wrong words?

| | Small, CI | Small int8 | Tiny |
|---|---|---|---|
| Wrong words in a passage on the review list | 51 / 71 | 44 / 64 | 156 / 165 |
| Correct words on the review list | 19 / 366 | 19 / 366 | 45 / 333 |
| Coloured words (probability < 0.5) that are wrong | — | 27 / 37 (73 %) | 100 / 127 (79 %) |
| Wrong words that are coloured | — | 27 / 64 (42 %) | 100 / 165 (61 %) |

The review list is reliable for **gross failures** and weak for **isolated
errors**: with Small int8, all 41 wrong words of the two babble items are
flagged, but of the 23 wrong words elsewhere (single word errors in clean
speech, French) only 3 are. A passage without a flag is not certified correct.

An extra detector was tried and rejected: flagging passages where the acoustic
model's own reading fits the audio clearly better than the transcript would
have found 4 more wrong words (83 instead of 79 of 87 on real English speech)
at the cost of flagging 68 more correct words.

### Acoustic evidence (pass 7)

On real speech (JFK, three LibriSpeech utterances), every single-word
substitution, insertion and repetition of the true text scored lower than
the truth, by at least 3.5 nats; deleting a short, quickly spoken word scored
up to 1.2 nats *better*, which is why the margin is 3 nats plus 3 nats per
dropped word. Speech the first pass missed entirely: from an empty first pass
over JFK, 3 of 4 speech stretches were recovered with margins above 3 nats
(WER 13.6 %); the fourth ("ask not what") is shorter than the 1 s minimum and
stays untranscribed. Known weak spot: on the last 2 s of the call (overlapping,
cut-off telephone speech that the reference annotation and VAD confirm), the
acoustic model hears only "oh" — such passages now get a medium doubt with an
explanation, not a high "possible hallucination".

### Word timestamps, interruption, speed

* Word starts on the synthetic conversations (exact reference times): mean
  absolute error 35–37 ms, 66–72 % within 50 ms (Small and Tiny).
* Interrupted processing: the two-person conversation was stopped mid-way (the
  app quitting), resumed (attempt 2) and produced text identical to an
  uninterrupted run with the same chunking (WER 0.0 with Small).
* Cost of re-checking (local, 4 vCPU): the 28 items took 185 s with one pass
  and 254 s re-checked (Small int8; 7 items had doubtful passages); 136 s and
  184 s with Tiny (16 items). Not measured on the target Mac.

### AI review with a local model

With `qwen2.5:0.5b` in Ollama (CI), a review of the call sent 10 passages in
one request (4 s). The model returned 3 suggestions; all 3 were rejected by
validation and stored with the reason — two referred to passages that were
not sent, one "punctuation" suggestion changed the words. Nothing was applied.
The model referred to passages by numbers that were not in its input; larger
models were not tested.
Accepting suggestions is tested with stored suggestions, not with a real
model's output.

## Defects found by testing and fixed

* An empty re-transcription could "replace" words (deleting them): empty
  decodes are now never replacements.
* Two doubtful passages in one window could both take the same decoded word
  (AMI trn08 duplicated "us."): each decoded word now belongs to one passage.
* Re-wordings that dropped words made AMI DER worse (CTC scoring favours
  dropping uncertain words): each dropped word now needs its own 3-nat lead,
  and passages overlapping another speaker are not re-worded.
* Re-checking a passage twice piled up flags and lost the original wording.
* An empty transcript over detected speech was reported as "No speech was
  detected" and never re-checked.
* "No acoustic support" was a high doubt even over speech other detectors
  confirmed (the end of the call).
* Rejecting an already accepted AI suggestion overwrote the decision.
* The schema migration re-added an existing column (found when a local
  benchmark failed mid-run).
* Benchmark flaws: DER was computed from segment spans (now word spans);
  interrupted runs were compared with differently chunked runs; empty
  transcripts were left out of DER.
* An older test expected every stage except speakers/alignment to complete
  (re-checking is skipped outside Maximum Accuracy).

## Known limitations

* Re-checking improves accuracy only modestly, and only where Whisper fails
  badly (babble). It cannot separate a foreground speaker from background
  speech, and it does not re-word overlapping passages.
* The acoustic model is English-only (Conformer-CTC); other languages need the
  MMS model (non-commercial licence). It does not recognise espeak's synthetic
  voice and is weak on far-field, overlapping and telephone speech.
* Review flags catch gross failures, not isolated word errors in clean speech.
* Maximum Accuracy's default model (Large-v3) was not part of the benchmark
  (Small and Tiny were); the benchmark measures the effect of the extra
  passes with the same model.
* AI review was tested only with a 0.5 B local model; OpenAI only through
  Ollama's OpenAI-compatible endpoint. Its usefulness is not measured.
* Speed and memory were measured on a 4-vCPU Linux container, not on a Mac.
* The corpus is small (28 items; about 420 reference words with a verified
  transcript, partly the same JFK speech under different conditions, partly
  synthetic) — the numbers describe this corpus, not all recordings.

## Launching

Download **Verbatim-app-unsigned** from the latest CI run's artifacts, or build
it on an Apple Silicon Mac (macOS 13+, Xcode 16, uv):

```sh
git checkout claude/sleepy-allen-06tojb
scripts/build_app.sh && open build/Verbatim.app
```

A CI build is not notarized: allow it once in System Settings › Privacy &
Security (or `xattr -dr com.apple.quarantine Verbatim.app`). Download a
Whisper model, the speaker model and the English alignment model in
**Settings › Models**; choose **Maximum Accuracy** to re-check doubtful
passages; use the **Review** menu in the transcript. AI review: install
Ollama, pull a model and choose it in **Settings › AI Verification**.

## Recommended next steps (Stage 4)

YouTube import of authorized media with yt-dlp (validated URLs, no DRM or
authentication bypass), captions where available (shown as a separate,
labelled source, never mixed silently into the transcript), full-text search,
SRT / VTT / JSON / CSV / Markdown export, and persistent processing queues. Carried over: measure Large-v3 and speed on a real Mac; number
passages per batch for small AI models.
