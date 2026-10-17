import XCTest
@testable import VerbatimCore

/// Decodes messages captured from a real engine run
/// (engine/scripts/capture_protocol_fixtures.py), so the Swift types are
/// checked against what the engine actually sends.
final class FixtureDecodingTests: XCTestCase {
    static let fixtures = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()   // VerbatimCoreTests
        .deletingLastPathComponent()   // Tests
        .deletingLastPathComponent()   // app
        .deletingLastPathComponent()   // repository root
        .appendingPathComponent("shared/protocol-fixtures")

    private struct Response<T: Decodable>: Decodable { let result: T }
    private struct Note<T: Decodable>: Decodable { let method: String; let params: T }
    private struct Failure: Decodable { let error: ErrorPayload }
    private struct FormatList: Decodable { let formats: [String] }

    private func data(_ name: String) throws -> Data {
        try Data(contentsOf: Self.fixtures.appendingPathComponent("\(name).json"))
    }

    private func result<T: Decodable>(_ name: String, as type: T.Type) throws -> T {
        try EngineClient.makeDecoder().decode(Response<T>.self, from: data(name)).result
    }

    func testEngineInfo() throws {
        let info = try result("engine.info", as: EngineInfo.self)
        XCTAssertEqual(info.protocolVersion, supportedProtocolVersion)
        XCTAssertEqual(Set(info.backends.map(\.name)), ["faster-whisper", "mlx"])
        let mlx = try XCTUnwrap(info.backends.first { $0.name == "mlx" })
        XCTAssertFalse(mlx.capabilities.beamSearch)
        XCTAssertTrue(info.backends.first { $0.name == "faster-whisper" }!.capabilities.beamSearch)
    }

    func testModesAndModels() throws {
        let modes = try result("transcription.modes", as: ModesResult.self).modes
        XCTAssertEqual(modes.map(\.id), ["fast", "balanced", "maximum"])
        XCTAssertFalse(modes.last!.limitations.isEmpty)
        XCTAssertEqual(modes.map(\.recheck), [false, false, true])

        let listing = try result("models.list", as: ModelsListing.self)
        XCTAssertEqual(listing.models.filter { $0.modelKind == "asr" }.count, 8)
        XCTAssertEqual(listing.models.filter { $0.modelKind == "diarization" }.count, 1)
        XCTAssertEqual(listing.models.filter { $0.modelKind == "alignment" }.count, 3)
        XCTAssertTrue(listing.models.allSatisfy { !$0.status.installed })
        let diarization = try XCTUnwrap(listing.models.first { $0.modelKind == "diarization" })
        XCTAssertNil(diarization.repoId)
        XCTAssertEqual(diarization.sourceDescription, "github.com")
        XCTAssertFalse(diarization.license?.isEmpty ?? true)
        let mms = try XCTUnwrap(listing.models.first { $0.id == "alignment/mms-300m" })
        XCTAssertTrue(mms.license?.contains("non-commercial") ?? false)
        let large = try XCTUnwrap(listing.models.first { $0.id == "mlx/large-v3" })
        XCTAssertEqual(large.repoId, "mlx-community/whisper-large-v3-mlx")
        XCTAssertGreaterThan(large.approxDownloadBytes, 2_000_000_000)
        XCTAssertNotNil(large.memoryEstimates["float16"])
        _ = try result("settings.get", as: EngineSettings.self)
    }

    func testProjectAndMedia() throws {
        let media = try result("media.probe", as: MediaInfo.self)
        XCTAssertEqual(media.audioStreams.first?.sampleRate, 16_000)
        XCTAssertFalse(media.hasVideo)
        let project = try result("projects.create", as: Project.self)
        XCTAssertEqual(project.title, "JFK")
        XCTAssertEqual(project.status, .new)
        XCTAssertEqual(project.duration ?? 0, 11.0, accuracy: 0.01)
        XCTAssertNotNil(project.updatedDate)
        let listed = try result("projects.list", as: ProjectsResult.self).projects
        XCTAssertEqual(listed.first?.status, .ready)
        XCTAssertGreaterThan(listed.first?.segmentCount ?? 0, 0)
    }

    func testJobsAndProgress() throws {
        let queued = try result("jobs.start", as: Job.self)
        XCTAssertEqual(queued.status, .queued)
        XCTAssertEqual(queued.kind, "transcribe")
        XCTAssertEqual(queued.stages.map(\.stage),
                       ["prepare", "extract", "analyze", "load_model", "transcribe", "diarize", "align",
                        "quality", "retranscribe", "validate", "finalize"])
        XCTAssertEqual(queued.options.diarize, .auto)

        // speaker and alignment models were not installed for this job, and
        // re-checking runs only in Maximum Accuracy mode
        let done = try result("jobs.get.completed", as: Job.self)
        XCTAssertEqual(done.status, .completed)
        let skipped = ["diarize", "align", "retranscribe"]
        XCTAssertTrue(done.stages.allSatisfy { $0.status == (skipped.contains($0.stage) ? .skipped : .completed) })

        let speakers = try result("jobs.get.completed.speakers", as: Job.self)
        XCTAssertTrue(speakers.stages.allSatisfy { $0.status == .completed })
        XCTAssertEqual(speakers.options.diarize, .on)
        XCTAssertEqual(speakers.effective?.lookup("diarization")?.lookup("speakers_found")?.doubleValue, 2)
        XCTAssertEqual(done.effective?.lookup("backend")?.stringValue, "faster-whisper")
        XCTAssertNil(done.error)

        let note = try EngineClient.makeDecoder().decode(Note<Job>.self, from: data("notification.job.updated.running"))
        XCTAssertEqual(note.method, "job.updated")
        let running = try XCTUnwrap(note.params.runningStage)
        // fraction is present only with a real denominator
        XCTAssertNotNil(running.fraction)
        XCTAssertEqual(running.progress?.unit, "seconds")
    }

    func testTranscriptAndExport() throws {
        let transcript = try result("transcript.get", as: Transcript.self)
        XCTAssertEqual(transcript.language, "en")
        XCTAssertFalse(transcript.segments.isEmpty)
        let first = transcript.segments[0]
        XCTAssertEqual(first.text, first.originalText)
        XCTAssertNil(first.speakerId)
        XCTAssertFalse(first.words?.isEmpty ?? true)
        let text = transcript.segments.map(\.text).joined(separator: " ")
        XCTAssertTrue(text.contains("fellow Americans"), text)
        XCTAssertEqual(transcript.effective?.lookup("model")?.lookup("display_name")?.stringValue != nil, true)

        let export = try result("export.render", as: ExportRendered.self)
        XCTAssertTrue(export.content.contains("[00:00:00]"))
        XCTAssertEqual(export.suggestedFilename, "JFK.txt")
    }

    func testSpeakersWordsAndCorrections() throws {
        let transcript = try result("transcript.get.speakers", as: Transcript.self)
        XCTAssertEqual(transcript.speakers.map(\.label), ["Speaker 1", "Speaker 2"])
        XCTAssertTrue(transcript.speakers.allSatisfy { $0.name == nil && $0.origin == "diarization" })
        XCTAssertTrue(transcript.segments.allSatisfy { $0.speakerId != nil && $0.speakerId == $0.originalSpeakerId })
        XCTAssertFalse(transcript.history.canUndo)
        let words = transcript.segments.flatMap { $0.words ?? [] }
        XCTAssertTrue(words.contains { $0.timingSource == "alignment" })
        XCTAssertTrue(words.allSatisfy { $0.speakerId != nil })
        XCTAssertTrue(transcript.segments.contains { $0.flagTypes.contains("split_at_speaker_change") })
        let starts = transcript.segments.map(\.start)
        XCTAssertEqual(starts, starts.sorted())

        let renamed = try result("transcript.renameSpeaker", as: EditResult.self)
        XCTAssertTrue(renamed.changed)
        XCTAssertEqual(renamed.transcript.speakers.first?.displayName, "Diane")
        XCTAssertEqual(renamed.transcript.speakers.first?.label, "Speaker 1")
        XCTAssertEqual(renamed.transcript.history.undoSummary, renamed.summary)

        let split = try result("transcript.splitSegment", as: EditResult.self)
        XCTAssertEqual(split.transcript.segments.count, transcript.segments.count + 1)
        let undone = try result("transcript.undo", as: EditResult.self)
        XCTAssertEqual(undone.transcript.segments.count, transcript.segments.count)
        XCTAssertTrue(undone.transcript.history.canRedo)
        XCTAssertGreaterThan(undone.transcript.revision, transcript.revision)

        let history = try result("transcript.history", as: EditHistoryResult.self).edits
        XCTAssertEqual(history.map(\.undone), [true, false])
        let original = try result("transcript.original", as: OriginalTranscript.self)
        XCTAssertEqual(original.segments.map(\.text), transcript.segments.map(\.originalText))
        XCTAssertEqual(original.speakers.map(\.label), ["Speaker 1", "Speaker 2"])

        let note = try EngineClient.makeDecoder().decode(Note<TranscriptUpdated>.self, from: data("notification.transcript.updated"))
        XCTAssertEqual(note.method, "transcript.updated")
        XCTAssertEqual(note.params.projectId, transcript.projectId)

        let export = try result("export.render", as: ExportRendered.self)
        XCTAssertFalse(export.content.isEmpty)
    }

    func testTristateEncoding() throws {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = .sortedKeys
        let options = JobOptions(mode: "fast", diarize: .auto, speakers: SpeakerBounds(num: 2), align: .off)
        let json = String(decoding: try encoder.encode(options), as: UTF8.self)
        XCTAssertTrue(json.contains("\"diarize\":\"auto\""), json)
        XCTAssertTrue(json.contains("\"align\":false"), json)
        XCTAssertTrue(json.contains("\"speakers\":{\"num\":2}"), json)
        let decoded = try EngineClient.makeDecoder().decode(JobOptions.self, from: Data(json.utf8))
        XCTAssertEqual(decoded, options)
    }

    func testStructuredErrors() throws {
        let decoder = EngineClient.makeDecoder()
        let missing = try decoder.decode(Failure.self, from: data("error.model_not_installed")).error
        XCTAssertEqual(missing.code, "model_not_installed")
        XCTAssertEqual(missing.data?.lookup("model_id")?.stringValue, "faster-whisper/large-v3")
        let notFound = try decoder.decode(Failure.self, from: data("error.not_found")).error
        XCTAssertEqual(notFound.code, "not_found")
        let speaker = try decoder.decode(Failure.self, from: data("error.speaker_not_found")).error
        XCTAssertEqual(speaker.code, "not_found")
    }

    func testPreflight() throws {
        let preflight = try result("transcription.preflight", as: Preflight.self)
        XCTAssertEqual(preflight.resolved.backend, "faster-whisper")
        XCTAssertEqual(preflight.options.mode, "balanced")
        XCTAssertEqual(preflight.resolved.diarization?.enabled, false)
        XCTAssertNotNil(preflight.resolved.diarization?.skipReason)
        XCTAssertEqual(preflight.resolved.recheck?.enabled, false)
        XCTAssertNotNil(preflight.resolved.recheck?.skipReason)
    }

    func testQualityReviewAndRecheck() throws {
        // Maximum Accuracy: quality checks, re-checking and validation ran
        let job = try result("jobs.get.completed.speakers", as: Job.self)
        XCTAssertEqual(job.mode, "maximum")
        XCTAssertEqual(job.effective?.lookup("recheck")?.lookup("enabled"), .bool(true))
        let validation = try XCTUnwrap(job.effective?.lookup("validation"))
        let transcript = try result("transcript.get.speakers", as: Transcript.self)
        XCTAssertEqual(Double(transcript.segments.filter(\.needsReview).count),
                       validation.lookup("needs_review")?.doubleValue)
        XCTAssertTrue(transcript.segments.allSatisfy { [nil, "low", "medium", "high"].contains($0.reviewPriority) })
        XCTAssertEqual(transcript.untranscribedSpeechCount, Int(validation.lookup("missing_speech")?.doubleValue ?? -1))

        // an on-demand re-check is a job of its own, saved as one undoable correction
        let queued = try result("transcript.recheck", as: Job.self)
        XCTAssertEqual(queued.kind, "recheck")
        XCTAssertEqual(queued.status, .queued)
        XCTAssertEqual(queued.stages.map(\.stage), ["prepare", "retranscribe", "finalize"])
        let done = try result("jobs.get.completed.recheck", as: Job.self)
        XCTAssertEqual(done.status, .completed)
        XCTAssertTrue(done.stages.allSatisfy { $0.status == .completed })
        let rechecked = try result("transcript.get.rechecked", as: Transcript.self)
        XCTAssertTrue(rechecked.history.undoSummary?.hasPrefix("Re-check 2 passages") ?? false)
        XCTAssertEqual(rechecked.segments.filter { $0.flagTypes.contains("rechecked") }.count, 2)

        let reviewed = try result("transcript.markReviewed", as: EditResult.self)
        let first = try XCTUnwrap(reviewed.transcript.segments.first)
        XCTAssertTrue(first.isReviewed)
        XCTAssertNil(first.reviewPriority)
        let removed = try result("transcript.deleteSegment", as: EditResult.self)
        XCTAssertEqual(removed.transcript.segments.count, rechecked.segments.count - 1)
        XCTAssertTrue(removed.transcript.history.canUndo)

        let missing = try EngineClient.makeDecoder().decode(Failure.self, from: data("error.recheck_segment_not_found")).error
        XCTAssertEqual(missing.code, "not_found")
    }

    /// An AI review with a small local model (Ollama), captured in CI.
    func testAIReviewIsLocalTraceableAndValidated() throws {
        let models = try result("llm.models", as: LLMModelsResult.self)
        XCTAssertEqual(models.provider, "ollama")
        XCTAssertFalse(models.remote)
        XCTAssertFalse(models.models.isEmpty)

        let started = try result("llm.review", as: LLMRun.self)
        XCTAssertTrue(started.isRunning)
        XCTAssertFalse(started.remote)
        let run = try result("llm.run", as: LLMRun.self)
        XCTAssertEqual(run.id, started.id)
        XCTAssertEqual(run.status, "completed")
        // every request and the model's answer are kept
        let exchanges = try XCTUnwrap(run.exchanges)
        XCTAssertEqual(exchanges.count, run.requests)
        XCTAssertTrue(exchanges.allSatisfy { $0.response != nil && $0.request.lookup("user")?.stringValue != nil })
        XCTAssertEqual(try result("llm.runs", as: LLMRunsResult.self).runs.first?.id, run.id)

        // suggestions are stored for review; the invalid ones say why
        let suggestions = try result("suggestions.list", as: SuggestionsResult.self).suggestions
        XCTAssertEqual(suggestions.count, run.suggestions?.values.reduce(0, +))
        XCTAssertTrue(suggestions.allSatisfy { ["pending", "invalid"].contains($0.status) && $0.runId == run.id })
        XCTAssertTrue(suggestions.filter { $0.status == "invalid" }.allSatisfy { !($0.validation ?? "").isEmpty })

        let note = try EngineClient.makeDecoder().decode(Note<LLMRun>.self, from: data("notification.llm.run.updated"))
        XCTAssertEqual(note.method, "llm.run.updated")
        XCTAssertEqual(note.params.id, run.id)

        // a remote endpoint is refused until the user allows it
        let refused = try EngineClient.makeDecoder().decode(Failure.self, from: data("error.llm_remote_not_allowed")).error
        XCTAssertEqual(refused.code, "invalid_params")
        XCTAssertEqual(refused.data?.lookup("remote"), .bool(true))
    }

    // MARK: - Stage 4

    func testAllExportFormats() throws {
        XCTAssertEqual(try result("export.formats", as: FormatList.self).formats, ExportFormat.allCases.map(\.rawValue))
        func rendered(_ format: ExportFormat) throws -> ExportRendered {
            let export = try result("export.render.\(format.rawValue)", as: ExportRendered.self)
            XCTAssertEqual(export.suggestedFilename, "JFK.\(format.fileExtension)")
            return export
        }
        XCTAssertTrue(try rendered(.srt).content.hasPrefix("1\n00:00:00,000 --> "))
        XCTAssertTrue(try rendered(.vtt).content.hasPrefix("WEBVTT\n"))
        // a byte-order mark, so spreadsheet apps read the text as UTF-8
        XCTAssertEqual(try rendered(.csv).content.unicodeScalars.first, "\u{FEFF}")
        XCTAssertTrue(try rendered(.md).content.hasPrefix("# JFK\n"))

        // the JSON export: metadata, speakers, segments with words and flags, and the original transcript
        let json = try JSONDecoder().decode(JSONValue.self, from: Data(try rendered(.json).content.utf8))
        XCTAssertEqual(json["format"]?.stringValue, "verbatim-transcript")
        XCTAssertEqual(json["project"]?["title"]?.stringValue, "JFK")
        guard case .array(let segments)? = json["segments"], let first = segments.first,
              case .array(let words)? = first["words"], case .array(let original)? = json["original"]?["segments"]
        else { return XCTFail("segments, words and the original transcript are expected") }
        XCTAssertFalse(words.isEmpty)
        XCTAssertEqual(original.count, segments.count)
        for key in ["speaker_id", "speaker", "start", "end", "text", "original_text", "flags", "review_priority"] {
            XCTAssertNotNil(first[key], key)
        }
        XCTAssertNotNil(json["speakers"])
    }

    func testSearch() throws {
        let found = try result("search.transcripts", as: SearchResponse.self)
        XCTAssertEqual(found.method, "index")
        XCTAssertEqual(found.terms, ["fellow", "americ"])
        let hit = try XCTUnwrap(found.results.first)
        XCTAssertEqual(hit.projectTitle, "JFK")
        XCTAssertFalse(hit.inOriginalOnly)
        // prefix matching: "americ" finds "Americans"
        XCTAssertEqual(hit.runs.filter { $0.1 }.map { $0.0 }, ["fellow", "Americans"])
        XCTAssertEqual(hit.runs.map { $0.0 }.joined(), hit.text)
        let empty = try EngineClient.makeDecoder().decode(Failure.self, from: data("error.search_empty")).error
        XCTAssertEqual(empty.code, "invalid_params")
    }

    func testPersistentQueue() throws {
        let paused = try result("queue.pause", as: QueueState.self)
        XCTAssertTrue(paused.paused)
        XCTAssertTrue(paused.autoResume)
        let waiting = try result("queue.get", as: QueueState.self)
        XCTAssertTrue(waiting.paused)
        XCTAssertNil(waiting.running)
        XCTAssertEqual(waiting.queued.map(\.queuePosition), [1, 2])
        XCTAssertTrue(waiting.queued.allSatisfy { $0.status == .queued })
        let reordered = try result("queue.reorder", as: QueueState.self)
        XCTAssertEqual(reordered.queued.map(\.id), waiting.queued.map(\.id).reversed())
        XCTAssertEqual(reordered.queued.map(\.queuePosition), [1, 2])
        let note = try EngineClient.makeDecoder().decode(Note<QueueUpdate>.self, from: data("notification.queue.updated"))
        XCTAssertEqual(note.method, "queue.updated")
        XCTAssertEqual(note.params.queued, reordered.queued.map(\.id))
        let resumed = try result("queue.resume", as: QueueState.self)
        XCTAssertFalse(resumed.paused)
    }

    /// Captured from a local web page with an audio file and creator captions
    /// (the same retrieval path as YouTube; YouTube itself is not reachable from CI).
    func testVideoImportWithCaptions() throws {
        let video = try result("youtube.inspect", as: YouTubeInspection.self)
        XCTAssertNil(video.refusal)
        XCTAssertTrue(video.adsNote.contains("not part of that file"))
        XCTAssertEqual(video.captions.map(\.id), ["manual:en"])
        XCTAssertFalse(video.captions[0].isAutomatic)

        let decoder = EngineClient.makeDecoder()
        let notVideo = try decoder.decode(Failure.self, from: data("error.youtube_bad_url")).error
        XCTAssertEqual(notVideo.code, "invalid_params")
        XCTAssertEqual(notVideo.data?.lookup("reason")?.stringValue, "url")
        let unconfirmed = try decoder.decode(Failure.self, from: data("error.youtube_not_authorized")).error
        XCTAssertEqual(unconfirmed.data?.lookup("reason")?.stringValue, "authorization")

        // the project exists before its media does
        let created = try result("projects.createFromYouTube", as: CreatedFromYouTube.self)
        XCTAssertNil(created.project.media)
        XCTAssertNil(created.project.sourceURL)
        XCTAssertTrue(created.project.isFromYouTube)
        XCTAssertEqual(created.job.kind, "youtube")
        XCTAssertEqual(created.job.stages.first?.stage, "download")
        XCTAssertEqual(created.job.stages.count, 12)

        let done = try result("jobs.get.completed.youtube", as: Job.self)
        XCTAssertEqual(done.status, .completed)
        let download = try XCTUnwrap(done.stages.first)
        XCTAssertEqual(download.status, .completed)
        XCTAssertEqual(download.progress?.unit, "bytes")
        XCTAssertEqual(download.progress?.done, download.progress?.total)
        let compared = done.effective?.lookup("source")?.lookup("captions_compared")
        guard case .array(let tracks)? = compared else { return XCTFail("captions were compared") }
        XCTAssertEqual(tracks.first?.lookup("language")?.stringValue, "en")

        let project = try result("projects.get.youtube", as: Project.self)
        XCTAssertEqual(project.source.kind, "url")
        XCTAssertNotNil(project.media)
        XCTAssertEqual(project.sourceURL?.path, project.media?.path)
        let meta = try XCTUnwrap(project.sourceMeta)
        XCTAssertTrue(meta.lookup("authorization")?.lookup("statement")?.stringValue?.hasPrefix("I am authorized") ?? false)
        XCTAssertNotNil(meta.lookup("authorization")?.lookup("confirmed_at")?.stringValue)
        XCTAssertEqual(meta.lookup("requested")?.lookup("media")?.stringValue, "audio")
        XCTAssertEqual(meta.lookup("retrieved")?.lookup("has_video"), .bool(false))

        let captions = try result("captions.list", as: CaptionsResult.self).captions
        XCTAssertEqual(captions.map(\.kind), ["manual"])
        XCTAssertEqual(captions.first?.projectId, project.id)
        XCTAssertEqual(captions.first?.cueCount, 2)
        let exported = try result("captions.export", as: ExportRendered.self)
        XCTAssertTrue(exported.content.hasPrefix("WEBVTT"))
        XCTAssertTrue(exported.suggestedFilename.hasSuffix(".en.captions.vtt"))
    }
}
