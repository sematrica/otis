# Engine protocol (version 4)

The app launches `verbatim-engine serve` (the embedded Python) and talks to it
over the child's stdin/stdout. Real captured messages for every shape below are
in `shared/protocol-fixtures/` and are decoded by the Swift tests.

## Framing

* One JSON object per line (UTF-8, `\n`-terminated, ASCII-escaped).
* Strict JSON only: a number that is not finite (NaN, infinity) is sent as
  `null` and a lone UTF-16 surrogate as U+FFFD; the engine logs where it
  replaced something. Every request gets a reply, also when its result could
  not be serialized (`internal_error`).
* When the engine cannot start, its last stderr line is
  `verbatim-engine: cannot start: <reason>` and it exits with status 3.
* stdout carries **only** protocol messages; logs go to stderr and
  `<data dir>/logs/engine.log`. The app must drain or redirect stderr.
* Requests may be answered out of order; match responses by `id`.

```
request       {"id": 7, "method": "projects.get", "params": {"project_id": "…"}}
success       {"id": 7, "result": {…}}
failure       {"id": 7, "error": {"code": "not_found", "message": "Project not found.", "data": {…}}}
notification  {"method": "job.updated", "params": {…}}          (no id; engine → app)
```

Closing stdin or calling `engine.shutdown` stops the engine. A running job is
interrupted (resumable), never silently lost.

## Methods

| Method | Params | Result |
|--------|--------|--------|
| `engine.info` | — | version, `protocol_version`, data dir, hardware, `backends[]` with capabilities |
| `engine.shutdown` | — | `{ok}`; the engine exits after replying |
| `media.probe` | `path` (absolute) | media info: format, duration, audio/video streams |
| `projects.create` | `path`, `title?` | project (validates and probes the file first) |
| `youtube.inspect` | `url` | what a link points to, before anything is downloaded (below); `invalid_params` (data `reason: url`) for anything but one YouTube video |
| `projects.createFromYouTube` | `url`, `authorized` (must be `true`), `options`, `media?` (`video` default, or `audio`), `captions?[{language, kind}]`, `title?` | `{project, job}`: the project (no media yet) and its queued `youtube` job. Refused without `authorized: true` (`invalid_params`, data `reason: authorization`), when the model is missing, or with the inspection's refusal code |
| `captions.list` | `project_id`, `include_cues?` | `{captions: [{id, project_id, language, name, kind (manual / auto), source, fetched_at, cue_count, cues?[{start, end, text}]}]}`: caption tracks as published, kept apart from the transcript |
| `captions.export` | `project_id`, `caption_id`, `format` (`srt` / `vtt`), `path?` | `{content, suggested_filename}`, or `{path, bytes}` with `path` |
| `projects.list` | `status?`, `limit?` | `{projects: [project]}` newest first |
| `projects.get` | `project_id` | project |
| `projects.rename` | `project_id`, `title` | project |
| `projects.delete` | `project_id` | `{deleted}`; cancels its job and removes working files |
| `transcription.modes` | — | `{modes: [{id, label, description, family, beam_size, best_of, cpu_compute_type, limitations, align, recheck}]}` |
| `transcription.preflight` | `options` | `{options, resolved: {mode, backend, model, chunk_seconds, warnings, diarization: {enabled, skip_reason, speakers}, alignment: {enabled, candidates, skip_reason}, recheck: {enabled, skip_reason, acoustic_models}}}` or `model_not_installed` |
| `jobs.start` | `project_id`, `options` | job (queued). Refused if the model is missing. |
| `jobs.cancel` / `jobs.resume` | `job_id` | job |
| `jobs.get` | `job_id` | job |
| `jobs.list` | `active_only?`, `project_id?`, `limit?` | `{jobs: [job]}` |
| `queue.get` | — | queue state: `{paused, auto_resume, running: job \| null, queued: [job]}` in the order they will run |
| `queue.pause` | `interrupt?` | queue state. No further job starts; with `interrupt`, the running job also stops now, keeps its progress and is queued again at the front |
| `queue.resume` | — | queue state |
| `queue.reorder` | `job_ids[]` (exactly the queued jobs, in the new order) | queue state; `invalid_job_state` if the list is not the queued set |
| `queue.setAutoResume` | `enabled` | queue state. When on (default), work interrupted by a quit or crash is queued again when the engine starts (at most 5 attempts per job) |
| `transcript.get` | `project_id`, `include_words?` | transcript (below) |
| `transcript.original` | `project_id` | the AI-generated transcript as first stored (speakers, segments, words); never changed by corrections |
| `transcript.history` | `project_id` | `{edits: [{id, kind, summary, undone, created_at}]}` newest first |
| `transcript.renameSpeaker` | `project_id`, `speaker_id`, `name` (null/"" = back to the label) | edit result |
| `transcript.createSpeaker` | `project_id`, `name?` | edit result (a user speaker without segments) |
| `transcript.mergeSpeakers` | `project_id`, `speaker_ids[]`, `into` | edit result; merged speakers move to `merged_speakers` |
| `transcript.assignSpeaker` | `project_id`, `segment_ids[]`, `speaker_id` **or** `new_speaker_name` | edit result |
| `transcript.splitSegment` | `project_id`, `segment_id`, `word_index` (first word of the new segment, ≥ 1) | edit result |
| `transcript.mergeSegments` | `project_id`, `segment_ids[]` (adjacent) | edit result |
| `transcript.editText` | `project_id`, `segment_id`, `text` | edit result (words re-derived, see below) |
| `transcript.markUncertain` | `project_id`, `segment_id`, `uncertain`, `note?` | edit result |
| `transcript.undo` / `transcript.redo` | `project_id` | edit result (`changed: false` when there is nothing to undo/redo) |
| `transcript.deleteSegment` | `project_id`, `segment_id` | edit result (the passage and its words are removed; undoable) |
| `transcript.chooseWording` | `project_id`, `segment_id`, `source` (`first_pass` or a candidate's `source`) | edit result: the wording recorded by re-checking is used and its flags are marked `resolved` |
| `transcript.markReviewed` | `project_id`, `segment_ids[]`, `reviewed?` (default true) | edit result: a `reviewed` flag is added (or removed) |
| `transcript.recheck` | `project_id`, `segment_ids[]` | job of kind `recheck` (queued): passes 6–7 on those passages with the settings and extracted audio of the transcription that produced the transcript; the outcome is one undoable correction. Refused (`invalid_job_state`) while a job runs or when that audio is gone |
| `llm.models` | `provider` (`ollama` / `openai`), `base_url?`, `api_key?` | `{provider, endpoint, remote, models[]}`; nothing from a transcript is sent |
| `llm.review` | `project_id`, `provider`, `model`, `base_url?`, `api_key?`, `allow_remote?` | review run (`status: running`); progress via `llm.run.updated`. An endpoint not on this computer is refused (`invalid_params`, data `remote: true`) unless `allow_remote` is true |
| `llm.runs` | `project_id` | `{runs: [run]}` newest first |
| `llm.run` | `run_id`, `exchanges?` | run; with `exchanges`, every request and response as sent and received |
| `llm.cancel` | `run_id` | run |
| `suggestions.list` | `project_id`, `statuses?[]` | `{suggestions: [suggestion]}` |
| `suggestions.accept` | `suggestion_id` | `{suggestion, summary, changed, transcript?, job}`: punctuation and terminology suggestions are applied as an undoable text edit, `suspicious` marks the passage uncertain with the model's reason, `retranscribe` starts a `recheck` job (the audio decides, not the language model). A suggestion whose passage changed since is marked `outdated` and not applied |
| `suggestions.reject` | `suggestion_id` | suggestion |
| `search.transcripts` | `query`, `project_id?`, `speaker_id?`, `include_original?`, `limit?` (default 100, at most 500) | `{query, terms[], method (index / substring), results[{project_id, project_title, segment_id, start, end, speaker_id, speaker, text, highlights[[start, end]], in_original_only}], truncated}`; `invalid_params` when the query has no words |
| `export.formats` | — | `{formats: ["txt", "srt", "vtt", "json", "csv", "md"]}` |
| `export.render` | `project_id`, `format`, `options?` | `{content, suggested_filename}` (formats below) |
| `export.write` | `project_id`, `format`, `path`, `options?` | `{path, bytes}` (atomic write) |
| `models.list` | — | `{models_dir, models: [model + status], downloads, backends}`; `model.kind` ∈ asr, diarization, alignment, with `license`, `languages` (empty = any), `homepage` |
| `models.download` | `model_id` | download state; progress via `model.download` notifications |
| `models.cancelDownload` | `model_id` | `{ok}` |
| `models.delete` | `model_id` | `{model_id, status}` |
| `settings.get` / `settings.set` | `models_dir?` (absolute path or null for default) | `{models_dir, default_models_dir}` |

### Search

The user's text is never passed to SQLite as query syntax: it is split into
words and `"quoted phrases"`; words match as prefixes (`americ` finds
"Americans"), all terms must occur, and case and diacritics are ignored
(FTS5, `unicode61 remove_diacritics 2`). Languages written without spaces
(Chinese, Japanese, Thai, …) are searched by substring (`method:
"substring"`). `highlights` are `[start, end)` offsets into `text` in Unicode
code points. With `include_original`, passages whose AI wording matched but
the user's correction does not are included with `in_original_only: true`.

### Export formats

| format | content | options |
|--------|---------|---------|
| `txt` | header (title, source, duration, language, model), then one block per speaker turn: `[hh:mm:ss] Speaker` and the text (per passage `[hh:mm:ss] text` without speakers) | `timestamps`, `header`, `speakers` (default true) |
| `srt` | subtitle cues cut from the word times: at most 2 lines of 42 characters and 7 s, ending at a sentence end once a cue has run 1.5 s, at least 0.7 s on screen (never into the next cue), never across two passages; the speaker's name starts the first cue of each turn | `speakers` |
| `vtt` | the same cues with `<v Speaker>` voice tags; text escaped | `speakers` |
| `json` | `{format: "verbatim-transcript", version: 1, exported_at, project {id, title, source, duration, language, created_at, updated_at, transcription (the producing job's mode, model, settings), revision}, speakers[{id, label, name, display_name, origin, …}], segments[{id, start, end, speaker_id, speaker, original_speaker_id, text, original_text, text_edited, speaker_edited, language, review_priority, flags[], words[{start, end, text, probability, timing_source, speaker_id}]}], original {job_id, created_at, speakers, segments}}`: the corrected transcript, every uncertainty flag, word times where available and the original AI transcript | — |
| `csv` | one row per passage (`segment_id, start, end, start_clock, speaker, text, original_text, text_edited, review_priority, open_flags`), or per word with `words: true`; UTF-8 with a byte-order mark; cells starting with `=`, `+`, `-`, `@`, tab or carriage return get a leading `'` so spreadsheets do not run them as formulas | `words`, `bom` |
| `md` | title, source and model, one paragraph per passage starting `**[hh:mm:ss] Speaker**`, and a "Passages to review" section listing open doubts | `review` (default true) |

### Video import (YouTube)

`youtube.inspect` result: `url` (canonical `https://www.youtube.com/watch?v=ID`),
`video_id, title, channel, channel_url, duration, upload_date, license,
language, live_status, availability, thumbnail, chapters[{start, end, title}],
sponsor_chapters[]` (chapters the creator titled as sponsor or ad segments),
`captions[{language, name, kind, formats}]` (`manual`: the creator's; `auto`:
YouTube's automatic captions in the video's own language, never machine
translations), `approx_bytes {video, audio}` (YouTube's estimates, null when
not given), `refusal {code, message} | null` and `ads_note`.

Accepted links: one video, as `youtube.com/watch?v=`, `youtu.be/`, `/shorts/`,
`/live/`, `/embed/`, `m.` and `music.` hosts and `youtube-nocookie.com/embed/`;
playlists, channels, other sites, credentials and ports in the URL are refused
before any network access. Retrieval uses yt-dlp with no cookies, no sign-in,
no geo bypass and no unplayable or DRM formats; only the YouTube extractor
runs. Media: AAC audio and, unless `media: "audio"`, H.264 video up to 720p,
joined without re-encoding (or a combined format when that is all there is),
stored under `<data dir>/media/<project id>/` and removed with the project.
For tests only, `VERBATIM_TEST_MEDIA_HOST` may name one loopback `host:port`
whose pages are accepted instead (yt-dlp's generic extractor).

A URL project's `source` is `{kind: "url", url, path, size_bytes, mtime_ns}`
with `path` null and `media` null until the media is retrieved;
`source_meta` records the video's metadata, the user's authorization
`{confirmed_at, statement}`, what was requested `{media, caption_tracks}`
and, after retrieval, `retrieved {file, bytes, formats, has_video, duration,
captions, warnings}`.

Ads: YouTube inserts most ads while a video plays; they are not in the
retrieved file and are not transcribed. Sponsor messages that are part of
the video are transcribed like any other speech; passages inside a sponsor
chapter get an informational `in_sponsor_chapter` flag.

### Job options

```
{ "mode": "fast" | "balanced" | "maximum",           default "balanced"
  "backend": "auto" | "faster-whisper" | "mlx",       default "auto"
  "model_id": "mlx/large-v3" | …,                     optional override
  "model_path": "/abs/path",                          custom model folder (requires backend)
  "language": "en" | … | null,                        null = detect
  "chunk_seconds": 20…3600,                           default 600
  "vad": true,
  "diarize": true | false | "auto",                   default "auto" (when the model is installed)
  "speakers": {"num": n | null, "min": n | null, "max": n | null},   all null = estimate
  "align": true | false | "auto",                     default "auto" (false in Fast mode)
  "alignment_model": "alignment/en-conformer-small" | …   optional; default by language
  "recheck": true | false | "auto" }                  default "auto" (on in Maximum Accuracy only)
```

`true` refuses to start (`model_not_installed`) when the model is missing;
`"auto"` skips the stage and records why (`effective.diarization.reason`,
`effective.alignment.reason`).

### Job snapshot

`kind` ∈ transcribe, recheck, youtube. `status` ∈ queued, running, completed, failed,
cancelled, interrupted. `stages[]` in order: for `transcribe` prepare, extract,
analyze, load_model, transcribe, diarize, align, quality, retranscribe,
validate, finalize; for `recheck` prepare, retranscribe, finalize; for
`youtube` download (`unit: "bytes"`; a total only when YouTube reported exact
sizes) followed by the `transcribe` stages. A resumed `youtube` job continues
a partial download and never retrieves finished media twice.
`queue_position` (1 = next) is set while the job is queued. Each has `status` (pending, running, completed, failed,
skipped, cancelled), `label`, `elapsed_seconds`, `progress {done,total,unit}|null`,
`fraction` (only when a real numerator and denominator exist, else null),
`eta_seconds` (only after enough measured work in the current attempt, else null)
and `detail`. `effective` records the settings that actually ran (backend,
model, revision, beam size, compute type, VAD, chunking, warnings) and, since
protocol 2, `diarization` (method, speakers found and in the transcript,
estimated count, overlap seconds, timings) and `alignment` (model, segments
aligned / failed / skipped, wildcard words); since protocol 3 `quality`
(speech detectors, flag counts), `recheck` (enabled or the reason it was not,
targets, windows, decodes, acoustic model, and counts: replaced, recovered,
unresolved, no_acoustic_support, confirmed, same_wording, not_comparable) and
since protocol 4 `source` for video
imports (`captions_compared[{language, kind, passages_differ}]`,
`sponsor_chapters`, `segments_in_sponsor_chapters`), and `validation` (segments, timing_fixes, words_out_of_order, possible_duplicates,
missing_speech, untranscribed_seconds, untranscribed[{start, end, severity}] (at
most 100 listed), review {high, medium, low}, needs_review). The
`retranscribe` stage counts decodes (`unit: "decodes"`); no stage reports a
fraction without a real denominator.

### Transcript

```
{ project_id, language, job_id, effective, revision,          revision: bumps on every change
  history: {can_undo, can_redo, undo_summary, redo_summary, edit_count},
  speakers: [speaker], merged_speakers: [speaker],
  segments: [segment] }                                       ordered by start time
```

`effective` is the producing job's, also for an empty transcript: there,
`effective.validation.missing_speech` says whether speech was detected that no
word covers (an empty transcript is not necessarily a silent recording).

Speaker: `id, label ("Speaker 1"), name (user-given or null), display_name,
origin (diarization | user), color_index, merged_into, segment_count,
speaking_seconds, first_start`. Labels are numbered by first appearance; no
speaker is ever given a real name automatically.

### Segment

`id, ordinal (position in the current transcript), start, end, text,
original_text, language, avg_logprob, no_speech_prob, compression_ratio,
chunk_index, speaker_id, original_speaker_id (the diarization's choice),
speaker_confidence (0–1 or null), text_edited, speaker_edited, flags[],
review_priority` (high, medium, low or null: the most severe open flag; a
`reviewed` flag clears it, except for the user's own uncertainty mark) and,
with `include_words`, `words[{start, end, text, probability, timing_source,
speaker_id}]`. `timing_source` ∈ whisper (recogniser estimate), alignment
(forced alignment), wildcard (timed by the aligner's wildcard: digits,
symbols), interpolated (a word a user added or rewrote).

`original_text` is the recogniser output for the segment's words and is never
changed by text edits; the complete AI transcript is also kept separately
(`transcript.original`).

Flags record every automatic decision and every doubt:

| type | meaning |
|------|---------|
| `chunk_boundary` | how a processing-chunk cut was joined (`method`: pause, aligned, time_fallback) |
| `trimmed_at_chunk_boundary` | words outside the chunk's owned span were dropped |
| `boundary_duplicate_removed` | a duplicated passage was removed (`text`, times) |
| `split_at_speaker_change` | a recognised segment was split where the speaker changes (`part`, `parts`, original times) |
| `speaker_uncertain` | `reasons`: low_coverage, windows_disagree, similar_to_other_speaker, no_speech_under_words; `coverage`, `agreement`, `margin` |
| `speaker_change_uncertain` | a word next to a speaker change may belong to the other side (`words`) |
| `overlapping_speech` | two or more people talk at once (`fraction`, `seconds`, `other_speakers` ids) |
| `speaker_unassigned` | diarization found no speech near the words |
| `alignment_failed` | forced alignment was rejected; recogniser timings kept (`reason`, `score`) |
| `marked_uncertain` | marked by the user (`note`); high |
| `low_confidence` | unclear speech (`reasons`: low_average_probability, many_uncertain_words, decoder_fallback, probability_unavailable (the recogniser's average log-probability is not a number; seen with MLX in repetition loops); `avg_logprob`, `uncertain_words` of `words`); `confirmed_acoustically` when re-checking found the audio clearly supports the wording |
| `possible_hallucination` | words that may not have been spoken (`reasons`: no_speech_detected, text_not_in_audio, repetition_loop, high_no_speech_probability, known_phrase (`phrase`), no_acoustic_support (`acoustic_margin`); `speech_coverage`). Words are never removed automatically |
| `repeated_words` | immediate repetitions (`repeats[{text, count, loop}]`); low (often genuine) |
| `missing_speech` | detected speech no word covers, on the segment before it (`start`, `end`, `detectors`: vad, diarization) |
| `retranscribed` | the passage was re-worded because another decode fits the audio better (`first_pass`: the recogniser's original wording, `chosen`, `margin` in nats, `candidates[{source, text, score}]`); low |
| `recovered_speech` | words added for untranscribed speech that beat "nothing was said" by the margin (`gap`, `chosen`, `margin`, `candidates`); medium |
| `alternatives` | other wordings within the margin, or not comparable (`candidates`, `margin`, `reason`); medium. The first-pass wording stays |
| `rechecked` | outcome of re-checking (`result`: confirmed, same_wording, unresolved, no_acoustic_support, not_comparable, not_rechecked (`reason`: overlapping speech), alternative_to_correction); informational |
| `possible_duplicate` | the start repeats the end of the previous segment (`text`) |
| `timing_inconsistent` | word times out of order (`words`) |
| `reviewed` | the user checked the passage (`by`, `at`); informational |
| `captions_differ` | the published captions say something clearly different here (`source`: manual or auto, `language`, `caption_text`, `agreement` 0–1); low. Captions are never merged into the transcript |
| `in_sponsor_chapter` | the passage lies in a chapter the creator titled as a sponsor or ad segment (`title`, `start`, `end`); informational |

Flags with `severity` use it; others default as listed. Any flag with
`resolved: true` (e.g. after `transcript.chooseWording`, with `chosen_by_user`)
no longer asks for review.

### Edit result

`{summary, changed, transcript}` where `transcript` includes words. Each
change is one undo step; a new change discards undone steps. Corrections are
refused with `invalid_job_state` while a job for the project is queued or
running (its result replaces the transcript). When a transcript is replaced by
a new transcription, its corrections and history are discarded.

Editing a segment's text re-derives its words: words that are still there
(ignoring case and punctuation) keep their timing, AI text and speaker; new or
rewritten words get interpolated times (`timing_source: interpolated`).

## Notifications

| Method | Params |
|--------|--------|
| `job.updated` | job snapshot (sent on every persisted change, ≤ 2/s during progress) |
| `project.updated` | project |
| `project.deleted` | `{project_id}` |
| `model.download` | `{model_id, status: running/cancelling/completed/failed/cancelled, done, total, total_is_estimate, error}` |
| `transcript.updated` | `{project_id, revision, summary}` after a correction, undo or redo |
| `llm.run.updated` | review run, on every change of status, batch progress or result |
| `queue.updated` | `{paused, auto_resume, running: job id \| null, queued: [job id]}` when the queue's order, paused state or contents change |

### AI review run and suggestion

Run: `id, project_id, provider, model, endpoint, remote, status (running,
completed, failed, cancelled), error, transcript_revision, segments_sent,
requests, started_at, finished_at, suggestions {status: count}` (and
`exchanges[{request, response, seconds, error}]` from `llm.run`). API keys are passed per
call by the app (from the macOS Keychain) and are never stored or logged.

Suggestion: `id, project_id, run_id, segment_id, kind (suspicious,
punctuation, terminology, retranscribe), original_text, suggested_text,
reason, status (pending, accepted, rejected, invalid, outdated), validation
(why a suggestion was invalid: e.g. a punctuation suggestion that changes
words), created_at, decided_at`. Nothing is applied until accepted.

## Error codes

`parse_error`, `invalid_request`, `method_not_found`, `invalid_params`,
`not_found`, `media_invalid`, `media_no_audio`, `source_changed`,
`model_not_installed` (data: `model_id`, `display_name`, `approx_download_bytes`),
`backend_unavailable`, `invalid_job_state`, `cancelled`, `download_failed`,
`worker_crashed`, `interrupted`, `internal_error`; since protocol 3
`audio_unavailable` (a re-check job found no extracted audio), `llm_error`
(the language-model endpoint failed or answered out of format) and
`invalid_state` (an AI review of the project is already running); since
protocol 4 `youtube_blocked` (YouTube asks this computer to sign in to prove it
is not a bot), `youtube_restricted` (private, members-only, age-restricted,
paid or not available in this country), `youtube_live` (live, upcoming or
still processing), `youtube_drm`, `youtube_unavailable`, `youtube_unsupported`
(no audio Verbatim can retrieve), `youtube_network` and `youtube_error`.
