# Engine protocol (version 1)

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
| `transcription.modes` | — | `{modes: [{id, label, description, family, beam_size, best_of, cpu_compute_type, limitations}]}` |
| `transcription.preflight` | `options` | `{options, resolved: {mode, backend, model, chunk_seconds, warnings}}` or `model_not_installed` |
| `jobs.start` | `project_id`, `options` | job (queued). Refused if the model is missing. |
| `jobs.cancel` / `jobs.resume` | `job_id` | job |
| `jobs.get` | `job_id` | job |
| `jobs.list` | `active_only?`, `project_id?`, `limit?` | `{jobs: [job]}` |
| `transcript.get` | `project_id`, `include_words?` | `{project_id, language, job_id, effective, segments: [segment]}` |
| `export.formats` | — | `{formats: ["txt"]}` |
| `export.render` | `project_id`, `format`, `options?` | `{content, suggested_filename}` |
| `export.write` | `project_id`, `format`, `path`, `options?` | `{path, bytes}` (atomic write) |
| `models.list` | — | `{models_dir, models: [model + status], downloads, backends}` |
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
  "vad": true }
```

### Job snapshot

`status` ∈ queued, running, completed, failed, cancelled, interrupted.
`stages[]` in order: prepare, extract, analyze, load_model, transcribe, finalize,
each with `status`, `label`, `elapsed_seconds`, `progress {done,total,unit}|null`,
`fraction` (only when a real numerator and denominator exist, else null),
`eta_seconds` (only after enough measured work in the current attempt, else null)
and `detail`. `effective` records the settings that actually ran (backend,
model, revision, beam size, compute type, VAD, chunking, warnings).

### Segment

`id, ordinal, start, end, text, original_text, language, avg_logprob,
no_speech_prob, compression_ratio, chunk_index, speaker_id (null in Stage 1),
flags[]` and, with `include_words`, `words[{start, end, text, probability,
timing_source}]`. `original_text` is the recogniser output and is never modified.

Flags record every automatic decision: `chunk_boundary` (`method`: pause,
aligned, time_fallback), `trimmed_at_chunk_boundary`, and
`boundary_duplicate_removed` (with the removed text and times).

## Notifications

| Method | Params |
|--------|--------|
| `job.updated` | job snapshot (sent on every persisted change, ≤ 2/s during progress) |
| `project.updated` | project |
| `project.deleted` | `{project_id}` |
| `model.download` | `{model_id, status: running/cancelling/completed/failed/cancelled, done, total, total_is_estimate, error}` |

## Error codes

`parse_error`, `invalid_request`, `method_not_found`, `invalid_params`,
`not_found`, `media_invalid`, `media_no_audio`, `source_changed`,
`model_not_installed` (data: `model_id`, `display_name`, `approx_download_bytes`),
`backend_unavailable`, `invalid_job_state`, `cancelled`, `download_failed`,
`worker_crashed`, `interrupted`, `internal_error`.
