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
        XCTAssertEqual(queued.stages.map(\.stage),
                       ["prepare", "extract", "analyze", "load_model", "transcribe", "diarize", "align", "finalize"])
        XCTAssertEqual(queued.options.diarize, .auto)

        // speaker and alignment models were not installed for this job
        let done = try result("jobs.get.completed", as: Job.self)
        XCTAssertEqual(done.status, .completed)
        XCTAssertTrue(done.stages.allSatisfy { $0.status == (["diarize", "align"].contains($0.stage) ? .skipped : .completed) })

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
    }
}
