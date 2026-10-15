import Foundation
import XCTest
@testable import VerbatimCore

/// Talks to the real Python engine exactly like the app does.
///
/// Requires `VERBATIM_ENGINE_PYTHON` (a Python with verbatim-engine installed).
/// The transcription test additionally needs `VERBATIM_TEST_MODEL_DIR`
/// (real Whisper weights) and optionally `VERBATIM_TEST_BACKEND`.
final class EngineIntegrationTests: XCTestCase {
    private var environment: [String: String] { ProcessInfo.processInfo.environment }

    private static let repositoryRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    private func startEngine() async throws -> (EngineClient, EngineInfo) {
        guard let python = environment[EngineLocator.environmentKey] else {
            throw XCTSkip("Set \(EngineLocator.environmentKey) to run engine integration tests")
        }
        let dataDir = FileManager.default.temporaryDirectory.appendingPathComponent("verbatim-tests-\(UUID().uuidString)")
        let client = EngineClient()
        let info = try await client.start(EngineLocator.configuration(python: URL(fileURLWithPath: python), dataDirectory: dataDir))
        return (client, info)
    }

    func testHandshakeErrorsAndProjects() async throws {
        let (client, info) = try await startEngine()
        XCTAssertEqual(info.protocolVersion, 1)
        XCTAssertTrue(info.backends.contains { $0.name == "faster-whisper" && $0.available })

        do {
            _ = try await client.project("does-not-exist")
            XCTFail("expected an error")
        } catch let error as EngineClientError {
            XCTAssertEqual(error.engineCode, "not_found")
        }
        do {
            _ = try await client.probe(path: "/definitely/not/here.mp4")
            XCTFail("expected an error")
        } catch let error as EngineClientError {
            XCTAssertEqual(error.engineCode, "media_invalid")
        }

        let jfk = Self.repositoryRoot.appendingPathComponent("engine/tests/corpus/jfk.wav")
        let project = try await client.createProject(path: jfk.path)
        XCTAssertEqual(project.media.audioStreams.first?.sampleRate, 16_000)
        let projects = try await client.listProjects()
        XCTAssertEqual(projects.map(\.id), [project.id])
        let renamed = try await client.renameProject(project.id, title: "Inaugural")
        XCTAssertEqual(renamed.title, "Inaugural")
        let models = try await client.models()
        XCTAssertEqual(models.models.count, 8)
        try await client.deleteProject(project.id)
        let afterDelete = try await client.listProjects()
        XCTAssertTrue(afterDelete.isEmpty)

        await client.stop()
        let running = await client.isRunning
        XCTAssertFalse(running)
    }

    func testRealTranscriptionThroughTheApp() async throws {
        guard let modelDir = environment["VERBATIM_TEST_MODEL_DIR"] else {
            throw XCTSkip("Set VERBATIM_TEST_MODEL_DIR (real Whisper weights) to run")
        }
        let (client, _) = try await startEngine()
        let backend = environment["VERBATIM_TEST_BACKEND"] ?? "faster-whisper"
        let jfk = Self.repositoryRoot.appendingPathComponent("engine/tests/corpus/jfk.wav")
        let project = try await client.createProject(path: jfk.path, title: "JFK")
        let job = try await client.startJob(projectId: project.id,
                                            options: JobOptions(mode: "balanced", backend: backend, modelPath: modelDir))
        let final = try await waitForJob(job.id, client: client, timeout: 600)
        XCTAssertEqual(final.status, .completed, final.error?.message ?? "")
        XCTAssertEqual(final.effective?.lookup("backend")?.stringValue, backend)

        let transcript = try await client.transcript(projectId: project.id, includeWords: true)
        let text = transcript.segments.map(\.text).joined(separator: " ")
        XCTAssertTrue(text.lowercased().contains("fellow americans"), text)
        XCTAssertEqual(transcript.language, "en")
        XCTAssertFalse(transcript.segments.first?.words?.isEmpty ?? true)

        let output = FileManager.default.temporaryDirectory.appendingPathComponent("verbatim-\(UUID().uuidString).txt")
        _ = try await client.writeExport(projectId: project.id, to: output.path)
        let exported = try String(contentsOf: output, encoding: .utf8)
        XCTAssertTrue(exported.contains("[00:00:00]"), exported)
        await client.stop()
    }

    func testEngineCrashIsReported() async throws {
        let (client, info) = try await startEngine()
        kill(pid_t(info.pid), SIGKILL)
        var sawTermination = false
        for await note in client.notifications {
            if case .engineTerminated = note {
                sawTermination = true
                break
            }
        }
        XCTAssertTrue(sawTermination)
        do {
            _ = try await client.info()
            XCTFail("expected notRunning")
        } catch let error as EngineClientError {
            XCTAssertEqual(error, .notRunning)
        }
    }

    private func waitForJob(_ id: String, client: EngineClient, timeout: TimeInterval) async throws -> Job {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            let job = try await client.job(id)
            if !job.status.isActive { return job }
            try await Task.sleep(for: .milliseconds(500))
        }
        throw EngineClientError.timedOut(method: "waiting for job \(id)")
    }
}
