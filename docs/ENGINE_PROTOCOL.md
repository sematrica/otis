# Engine protocol (version 2)

The app launches `verbatim-engine serve` (the embedded Python) and talks to it
over the child's stdin/stdout. Real captured messages for every shape below are
in `shared/protocol-fixtures/` and are decoded by the Swift tests.

## Framing

* One JSON object per line (UTF-8, `\n`-terminated, ASCII-escaped).
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
| `projects.list` | `status?`, `limit?` | `{projects: [project]}` newest first |
| `projects.get` | `project_id` | project |
| `projects.rename` | `project_id`, `title` | project |
| `projects.delete` | `project_id` | `{deleted}`; cancels its job and removes working files |
| `transcription.modes` | — | `{modes: [{id, label, description, family, beam_size, best_of, cpu_compute_type, limitations, align}]}` |
| `transcription.preflight` | `options` | `{options, resolved: {mode, backend, model, chunk_seconds, warnings, diarization: {enabled, skip_reason, speakers}, alignment: {enabled, candidates, skip_reason}}}` or `model_not_installed` |
| `jobs.start` | `project_id`, `options` | job (queued). Refused if the model is missing. |
| `jobs.cancel` / `jobs.resume` | `job_id` | job |
| `jobs.get` | `job_id` | job |
| `jobs.list` | `active_only?`, `project_id?`, `limit?` | `{jobs: [job]}` |
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
| `export.formats` | — | `{formats: ["txt"]}` |
| `export.render` | `project_id`, `format`, `options?` | `{content, suggested_filename}` |
| `export.write` | `project_id`, `format`, `path`, `options?` | `{path, bytes}` (atomic write) |
| `models.list` | — | `{models_dir, models: [model + status], downloads, backends}`; `model.kind` ∈ asr, diarization, alignment, with `license`, `languages` (empty = any), `homepage` |
| `models.download` | `model_id` | download state; progress via `model.download` notifications |
| `models.cancelDownload` | `model_id` | `{ok}` |
| `models.delete` | `model_id` | `{model_id, status}` |
| `settings.get` / `settings.set` | `models_dir?` (absolute path or null for default) | `{models_dir, default_models_dir}` |

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
  "alignment_model": "alignment/en-conformer-small" | …   optional; default by language }
```

`true` refuses to start (`model_not_installed`) when the model is missing;
`"auto"` skips the stage and records why (`effective.diarization.reason`,
`effective.alignment.reason`).

### Job snapshot

`status` ∈ queued, running, completed, failed, cancelled, interrupted.
`stages[]` in order: prepare, extract, analyze, load_model, transcribe, diarize,
align, finalize, each with `status` (pending, running, completed, failed,
skipped, cancelled), `label`, `elapsed_seconds`, `progress {done,total,unit}|null`,
`fraction` (only when a real numerator and denominator exist, else null),
`eta_seconds` (only after enough measured work in the current attempt, else null)
and `detail`. `effective` records the settings that actually ran (backend,
model, revision, beam size, compute type, VAD, chunking, warnings) and, since
protocol 2, `diarization` (method, speakers found and in the transcript,
estimated count, overlap seconds, timings) and `alignment` (model, segments
aligned / failed / skipped, wildcard words).

### Transcript

```
{ project_id, language, job_id, effective, revision,          revision: bumps on every change
  history: {can_undo, can_redo, undo_summary, redo_summary, edit_count},
  speakers: [speaker], merged_speakers: [speaker],
  segments: [segment] }                                       ordered by start time
```

Speaker: `id, label ("Speaker 1"), name (user-given or null), display_name,
origin (diarization | user), color_index, merged_into, segment_count,
speaking_seconds, first_start`. Labels are numbered by first appearance; no
speaker is ever given a real name automatically.

### Segment

`id, ordinal (position in the current transcript), start, end, text,
original_text, language, avg_logprob, no_speech_prob, compression_ratio,
chunk_index, speaker_id, original_speaker_id (the diarization's choice),
speaker_confidence (0–1 or null), text_edited, speaker_edited, flags[]` and,
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
| `marked_uncertain` | marked by the user (`note`) |

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

## Error codes

`parse_error`, `invalid_request`, `method_not_found`, `invalid_params`,
`not_found`, `media_invalid`, `media_no_audio`, `source_changed`,
`model_not_installed` (data: `model_id`, `display_name`, `approx_download_bytes`),
`backend_unavailable`, `invalid_job_state`, `cancelled`, `download_failed`,
`worker_crashed`, `interrupted`, `internal_error`.
