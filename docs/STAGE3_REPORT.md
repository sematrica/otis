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

<!-- sections below are filled in from measured runs -->
