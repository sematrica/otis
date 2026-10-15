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
        XCTAssertEqual(listing.models.count, 8)
        XCTAssertTrue(listing.models.allSatisfy { !$0.status.installed })
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
        XCTAssertEqual(queued.stages.map(\.stage), ["prepare", "extract", "analyze", "load_model", "transcribe", "finalize"])

        let done = try result("jobs.get.completed", as: Job.self)
        XCTAssertEqual(done.status, .completed)
        XCTAssertTrue(done.stages.allSatisfy { $0.status == .completed })
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

    func testStructuredErrors() throws {
        let decoder = EngineClient.makeDecoder()
        let missing = try decoder.decode(Failure.self, from: data("error.model_not_installed")).error
        XCTAssertEqual(missing.code, "model_not_installed")
        XCTAssertEqual(missing.data?.lookup("model_id")?.stringValue, "faster-whisper/large-v3")
        let notFound = try decoder.decode(Failure.self, from: data("error.not_found")).error
        XCTAssertEqual(notFound.code, "not_found")
    }

    func testPreflight() throws {
        let preflight = try result("transcription.preflight", as: Preflight.self)
        XCTAssertEqual(preflight.resolved.backend, "faster-whisper")
        XCTAssertEqual(preflight.options.mode, "balanced")
    }
}
