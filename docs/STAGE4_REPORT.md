# Stage 4 (Media import, search, exports, queue) — report

All results below come from real runs: the real engine, real yt-dlp, real
models and real audio. Nothing is mocked. Where something was not verified,
it says so. Retrieval from YouTube itself could not be run from the
development container (it has no route to YouTube); what CI found is
reported under "YouTube itself".

## Implemented

| Requirement | Status |
|-------------|--------|
| 1. Validate the URL | Only one YouTube video is accepted (`watch?v=`, `youtu.be/`, `/shorts/`, `/live/`, `/embed/`, mobile and music hosts, `youtube-nocookie.com/embed/`) and turned into the canonical `https://www.youtube.com/watch?v=ID` before any network access. Other sites, playlists, channels, other schemes, user names, passwords and ports are refused with a message that says what to paste instead. |
| 2. Retrieve available metadata | **Check Link…** (or Return in the link field) shows the title, channel, length, upload date, licence as YouTube states it, chapters and the approximate size of video+audio and audio only. Nothing is downloaded. |
| 3. Determine whether captions are accessible | The creator's caption tracks and YouTube's automatic captions *in the video's own language* are listed (never machine translations). The creator's are chosen by default; automatic ones on request. |
| 4. Retrieve permitted media when supported | Only after the user ticks "I am authorized to download and transcribe this video: it is my own, it is licensed for reuse, or I have the rights holder's permission." The engine refuses without it, and stores the statement and time with the project. yt-dlp (pinned) runs with no cookies, no account, no `.netrc`, no geo bypass, no unplayable or DRM formats, only its YouTube extractor, and no scripts fetched at run time (YouTube's player challenges are solved by the pinned yt-dlp-ejs scripts in the bundled deno). Live, upcoming, account-only (private, members-only, paid, age-restricted), region-blocked and DRM videos are refused with the reason; when YouTube asks the computer to sign in ("not a bot"), the user is told that Verbatim does not sign in. AAC audio and, if wanted, H.264 video up to 720p are downloaded separately and joined without re-encoding; a combined format is used when that is all there is. Downloads continue after an interruption; finished media is never retrieved twice. Download progress shows a total only when YouTube reports exact sizes. |
| 5. Process audio through the pipeline | The `youtube` job's first stage ("Retrieving from YouTube") is followed by the full pipeline of Stages 1–3 (all modes, speakers, alignment, quality checks, re-checking, validation). The project's player plays the retrieved file. |
| 6. Speaker-labelled transcripts | As for local files. |
| Captions | Stored as published, separately from the transcript, and exportable as SRT or WebVTT ("Published captions (not the transcript)"). After transcription they are compared with the transcript; a passage where they clearly differ gets a low-severity "differs from the captions" badge that shows the caption wording. Captions never replace, add or remove transcript words. |
| Advertisements | Before import and in the project's details: "Ads that YouTube inserts while a video plays are not part of that file and are not transcribed. Sponsor messages or ads that are part of the video itself are transcribed like any other speech." Passages inside chapters the creator titled as sponsor or ad segments get an informational "sponsor chapter" badge. |
| Local-file fallback | When YouTube refuses (sign-in check, restrictions, region) the message says to download the video where the user may do so and import the file; importing local files is unchanged. |
| Full-text search | SQLite FTS5 over every passage, kept in step with corrections, undo and new transcripts by triggers. A **Search** section searches the whole library as you type (option: also find the AI's original wording of passages you corrected; the engine can also narrow to one project or speaker); ⌘F searches the open transcript. Words match as prefixes, "quoted phrases" exactly; case and diacritics are ignored; matches are highlighted; a result opens the project at the passage with the player cued to it. The user's text is never interpreted as query syntax. Languages written without spaces fall back to substring search. |
| Export formats | TXT, SRT, WebVTT, JSON, CSV and Markdown from the Export menu (and `verbatim-engine transcribe --format`). Subtitle cues are cut from the word times (at most two lines of 42 characters and 7 s, ending at a sentence end once a cue has run 1.5 s, never across speakers; the speaker's name starts each turn; VTT voice tags). JSON contains project metadata, speaker IDs and names, segment and word timestamps (with their source: alignment or recogniser), the corrected and the original transcript, and every uncertainty flag with the review priority. CSV (per passage or per word) is UTF-8 with a byte-order mark and defused against formula injection. Markdown has speaker turns and a "Passages to review" section. |
| Persistent processing queue | Jobs wait in a stored order; the Processing view shows "#n in the queue", moves jobs up and down, and pauses after the current job or immediately (the job keeps its finished chunks and goes back to the front). The paused state survives a restart. Work interrupted by a quit or crash is queued again when Verbatim starts (Settings › Processing queue; on by default; at most 5 automatic attempts per job). |
| Protocol | Version 4 (`docs/ENGINE_PROTOCOL.md`); database schema 6. Fixtures captured from a real engine run are decoded by the Swift tests. |

Out of scope (Stage 5): notarization and signing for distribution, updates
(including of yt-dlp), bundle-size reduction, performance tuning on real Macs.

## Verification

| Suite | Result |
|-------|--------|
| Engine unit, storage and RPC tests (no models) | 197 passed (Linux); 196 passed, 1 skipped (macOS) |
| Engine real-model tests | Linux: 32 passed, 5 skipped (3 MLX tests need Apple Silicon; the real-YouTube test needs its variable, set in a separate step; in the AI-review test the model made no applicable suggestion, so the accept step was skipped). macOS: 32 passed, 5 skipped (MMS model, Ollama, real YouTube) |
| Real YouTube (separate, non-blocking step) | skipped: YouTube refused the runner (below) |
| Swift tests (protocol fixtures, real engine, real transcription and corrections) | 26, 0 failures (15 decode captured messages, 4 new for Stage 4) |
| Packaged-app self-tests on macOS | faster-whisper with exports, search, queue and a page import through the bundled yt-dlp; MLX; Maximum Accuracy with speakers and alignment on the two-person call (80 / 80 words aligned, rename, undo, on-demand re-check) |
| Accuracy benchmark | runs in CI on every push (Whisper Small) |

Final Stage 4 CI run, all green on Linux and Apple Silicon macOS:
https://github.com/sematrica/otis/actions/runs/39045746835

New Stage 4 tests: the URL rules (every accepted form, every refused one,
the loopback-only test host); WebVTT parsing of creator captions and of
YouTube's rolling automatic captions; caption comparison; format choice
(AAC + H.264 ≤ 720p, combined fallback, DRM, nothing retrievable);
refusals, sponsor chapters and error explanations; real yt-dlp retrieval of
media and captions from a local web page, cancelling a download, and joining
streams without re-encoding; the whole import through the real worker with a
real model (inspect, refuse without authorization, retrieve, transcribe,
compare with captions, search, export, transcribe again, delete the project
and its media); subtitle cue limits, VTT escaping, the JSON contents the
specification lists, CSV injection, Markdown; search (prefixes, diacritics,
phrases, filters, query syntax never interpreted, the index after
corrections, undo and new transcripts, unspaced scripts, limits); and the
queue (order, reorder, pause, pause now and continue, auto-resume after a
restart, turning it off, persisted state).

## Measured results

### YouTube itself

CI tried it on GitHub's Linux runner (run 27,
https://github.com/sematrica/otis/actions/runs/39003082184) with the default
Creative Commons video. YouTube refused already when asked for the video's
metadata:

```
SKIPPED engine/tests/test_stage4_models.py:101: YouTube did not serve this machine: youtube_blocked:
YouTube refused the request from this computer and asks to sign in to prove it is not a bot. Verbatim
does not sign in to YouTube. Try again later, or download the video …
```

So: the refusal path (YouTube's bot check, explained to the user, nothing
bypassed) is verified against YouTube itself; **retrieving media and
captions from YouTube is not verified** — neither from the development
container (no route) nor from CI (refused). The retrieval code is the same
as for the web page below except for yt-dlp's YouTube extractor and the
player-challenge solving in deno, which therefore remain untested here. The
test can be repeated on any machine YouTube serves, with
`VERBATIM_TEST_YOUTUBE_URL` (and `VERBATIM_TEST_YOUTUBE_AUTHORIZED=1` for a
video that is not Creative Commons but that the person running it may
retrieve).

### Import from a web page (the same retrieval path, verified)

Real yt-dlp (its generic extractor; YouTube's extractor needs YouTube)
retrieving a local page with the JFK recording and a creator caption track,
through the real worker, with real Whisper:

| Check | Result |
|---|---|
| Metadata (`youtube.inspect`) | title "JFK talk (1)" (as yt-dlp's generic extractor names media found on a page), one creator caption track `en` (WebVTT), no refusal, the ads note |
| Refused without the authorization | `invalid_params`, reason `authorization` |
| Retrieval (audio only) | 352,078 bytes (the WAV as served), download stage 352,078 / 352,078 bytes |
| Retrieval (video and audio, model test) | an MP4 with both streams, joined without re-encoding |
| Captions | 2 cues stored, exported as published (SRT and WebVTT) |
| Transcript (Whisper Small, int8) | "And so my fellow Americans, ask not what your country can do for you. Ask what you can do for your country." |
| Compared with the captions | 0 passages differ |
| Searched, exported, transcribed again, deleted | the project and its retrieved media are removed |

The packaged app did the same on the Apple Silicon runner (runs 28 and 29): its
self-test went through the app's own model layer and client — rendered all
six export formats of a new transcript (TXT 252, SRT 207, VTT 225, JSON
12,147, CSV 374, Markdown 255 characters), found it by searching
"Americans", read the queue, then inspected and imported a local page with
the bundled yt-dlp (`download` and every pipeline stage completed or
skipped as expected, the creator captions stored, transcript "And so my
fellow Americans, ask not what your country can do for you. Ask what you
can do for your country."). The bundle check runs the bundled tools:
`yt-dlp 2026.08.19 | deno 2.9.7 (stable, release, aarch64-apple-darwin)`.
The bundle is about 906 MB (Stage 1: about 800 MB; the deno binary alone is
about 96 MB on Linux).

### Search speed

`engine/scripts/benchmark_search.py --projects 100 --passages 2000`:
200,000 passages (6–18 made-up words each from a 20,000-word
Zipf-distributed vocabulary; at about 1,000 passages an hour, some 200 hours
of transcripts), stored and indexed through the engine (database 797 MB,
mostly word timings), 4-vCPU Linux container, 7 runs per query through
`search.transcripts` (the app asks for at most 300 results):

| Query | Results | Median | Max |
|---|---|---|---|
| rare word | 30 | 0.7 ms | 1.6 ms |
| mid-frequency word | 100 (limit) | 9.9 ms | 10.0 ms |
| two words | 100 (limit) | 20.0 ms | 20.9 ms |
| quoted phrase | 100 (limit) | 34.3 ms | 37.8 ms |
| three-letter prefix | 100 (limit) | 43.3 ms | 69.6 ms |
| most common word, one project | 100 (limit) | 62.0 ms | 77.4 ms |
| most common word (in most passages) | 100 (limit) | 266.6 ms | 354.1 ms |

An earlier run of the same measurement gave 10–30 % lower times (the
container is shared). Ranking every match is what makes a word that occurs
in most passages slow; the app searches as the user types, 0.3 s after
the last key. Building the library (storing 200,000 passages
with their words and indexing them) took 92 s.

### Subtitles from real transcripts

From the captured transcripts (Whisper Small int8; the call with forced
alignment):

| Transcript | Words (timing) | Cues | Longest line | Lines per cue | Cue length | Overlapping cues | Cues starting a speaker turn |
|---|---|---|---|---|---|---|---|
| JFK, one speaker | 22 (recogniser) | 3 | 37 chars | ≤ 2 | 0.70–6.85 s | 0 | — |
| Two-person call | 80 (alignment) | 13 | 40 chars | ≤ 2 | 0.24–4.12 s | 0 | 8 |

Cues shorter than 0.7 s occur where the next cue follows at once: a cue is
held for 0.7 s only when that does not run into the next one.

### Transcription unchanged

Stage 4 does not change transcription. In CI run 27 (Whisper Small, all
28 corpus items) the accuracy benchmark gave exactly the results of the final
Stage 3 run: WER single speaker 3.03 %, noise 57.95 % → 56.82 % with
re-checking, overlap 25.77 % → 25.26 %, silence 0 %, two-person 0 %,
multi-speaker 1.33 %, multilingual 33.85 %; 51 of 71 wrong words flagged,
19 of 366 right words flagged; DER call 0.015, AMI excerpts 0.08–0.71 apart
from the two without detectable speech, synthetic conversations 0.000.

Run 29, two commits later and with no change to transcription code (the
two engine changes in between concern the queue's state and the AI review's
answer length), gave different results on some items: German synthetic
speech 7.7 % → 169.2 % WER (words inserted), babble at 5 dB 63.6 % → 77.3 %,
babble at 0 dB 159.1 % → 145.4 %, AMI DER changes of up to 0.05
(dev00 0.445 → 0.485, trn04 0.333 → 0.283). The category totals for noise,
overlap, single and multi-speaker happened to be equal; multilingual went
from 33.85 % to 66.15 %. So **the benchmark is not reproducible bit for bit
across CI machines**: the same model, settings and audio decode differently
on some borderline inputs, most likely because GitHub's runners differ in
CPU (the engine does not record the CPU model yet). Single CI runs, including
the Stage 3 figures, describe one machine; differences of this size between
runs are not evidence of a change.

## Defects found by testing and fixed

* SRT lines exceeded 42 characters when the speaker's name was added to
  every cue: the name now starts only the first cue of a turn and counts
  toward the line length.
* Word times spread over a segment without word timings drifted past the
  segment's end (floating point): the last word now ends at the segment end.
* YouTube's rolling automatic captions lost their first cue: cue blocks were
  split at lines holding only a space, which YouTube puts in every cue.
* The "not available in your country" refusal was reported as a generic
  error.
* The captions comparison was computed but not stored with the job.
* A web page's audio without codec details was taken for video and the
  video then reported as DRM-protected: such formats are now audio, and DRM
  is reported only when a DRM format exists.
* The finished download stage showed the last throttled progress sample
  (1,024 of 352,078 bytes): it now records the retrieved size.
* Resuming interrupted work at startup raced with a manual resume in the
  restart test and the benchmark's interrupted run: both now leave it to the
  queue.
* A project waiting for its media could not be decoded by the app (media was
  required): media is optional until retrieved.
* The app did not build because a public type built by the app had no public
  initializer (found by CI).
* The queue named a finished job as running for a moment: the worker saves
  the outcome just before it exits, and the manager noticed the exit only
  afterwards (found by a Swift fixture test on CI).
* The MLX app self-test tried to import an empty page link (an empty
  variable was taken for a page).
* While CI captured fixtures, the 0.5 B local model repeated itself until
  the 300 s request timeout (constrained JSON output, no length limit):
  Ollama answers are now limited to 4,096 tokens and reported as cut off
  (Stage 3 component).

## Known limitations

* **YouTube itself was not tested from the development container**, which
  cannot reach YouTube; see "YouTube itself" for what CI found. YouTube
  often asks data-centre (and some home) addresses to sign in; Verbatim
  never signs in, so such imports fail with an explanation and the video has
  to be downloaded where the user may do so and imported as a file.
* yt-dlp is pinned and bundled. YouTube changes break older yt-dlp versions,
  usually within weeks to months; updating it needs a new app build (Stage 5).
* Whether the user is authorized is the user's statement; Verbatim cannot
  check it.
* Ads: YouTube's playback ads are not in the retrieved file and are not
  transcribed. Sponsor messages inside the video are transcribed; they are
  labelled only when the creator put them in a chapter whose title says so
  (English, German, Spanish and French words). Unmarked sponsor messages are
  not detected.
* The captions comparison is a heuristic (word-sequence agreement below 0.6
  over passages of at least three words); it was tested on synthetic cases
  and a local page, not tuned on real YouTube captions. Only captions in
  the transcript's language are compared; only WebVTT tracks are retrieved.
* Video is limited to H.264 up to 720p (or a combined format); videos that
  YouTube offers only in other codecs are imported as audio, or as a
  combined lower-resolution file.
* Search finds words by prefix: no stemming, spelling tolerance or synonyms.
  Speaker names and captions are not searched. Substring search for
  unspaced scripts is not ranked.
* Export layouts are fixed (no templates); the JSON format is documented in
  `docs/ENGINE_PROTOCOL.md`, not as a JSON Schema file.
* One job runs at a time (by design: models must not share memory on 16 GB).
* The bundled deno adds about 90–100 MB; its JIT will need the matching
  hardened-runtime entitlement for notarization (Stage 5).
* The new views have no automated UI tests; they are exercised through the
  app's model layer in the packaged-app self-test.
* The AI review's run in the CI fixture capture fails with the 0.5 B test
  model, which repeats itself (inventing passage numbers it was not sent)
  until the answer is cut off; the review fixtures decoded by the Swift
  tests are the ones CI captured for Stage 3, when the same model completed
  the review. Larger local models were not tested.
* Benchmark results vary between CI machines (see "Transcription
  unchanged").

## Launching

Download **Verbatim-app-unsigned** from the latest CI run's artifacts, or build
it on an Apple Silicon Mac (macOS 13+, Xcode 16, uv):

```sh
git checkout claude/sleepy-allen-06tojb
scripts/build_app.sh && open build/Verbatim.app
```

A CI build is not notarized: allow it once in System Settings › Privacy &
Security (or `xattr -dr com.apple.quarantine Verbatim.app`). In the Library,
paste a YouTube link into "Or paste a YouTube link…" and press **Check
Link…**; confirm you are authorized, choose video or audio and the captions,
then **Retrieve and Transcribe**. **Search** is in the sidebar (⌘F inside a
transcript); **Export** in the transcript's toolbar; the queue in
**Processing**.

## Recommended next steps (Stage 5)

Production readiness: signing with the hardened runtime and notarization
(including deno's JIT entitlement and every embedded library), an update
path for the app and for yt-dlp, bundle-size reduction, the FFmpeg licence
check of the bundled PyAV build, crash reporting that stays local, speed and
memory measured on real Macs (8 GB and 16 GB, Large-v3), accessibility, and
UI automation for the main flows. Also: record the CPU model with every
benchmark and repeat CI benchmark runs to measure their spread, and test
YouTube retrieval from a machine YouTube serves (a Mac at home or work) with
a video the tester may retrieve.
