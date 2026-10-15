import AppKit
import Foundation
import VerbatimCore

/// Non-interactive end-to-end check of the packaged app, enabled only by
/// environment variables (used by CI):
///
///     VERBATIM_SELF_TEST_MEDIA       file to import
///     VERBATIM_SELF_TEST_MODEL_DIR   real model folder
///     VERBATIM_SELF_TEST_BACKEND     faster-whisper (default) or mlx
///     VERBATIM_SELF_TEST_OUTPUT      folder for result.json and window.png
///
/// It goes through the same AppModel the UI uses (engine start, validation,
/// import, job notifications, transcript loading), opens the transcript
/// workspace, saves a snapshot of the window, then quits normally.
@MainActor
struct SelfTest {
    let media: URL
    let modelDirectory: String
    let backend: String
    let output: URL

    struct Failure: Error, CustomStringConvertible {
        let description: String
    }

    static func fromEnvironment() -> SelfTest? {
        let env = ProcessInfo.processInfo.environment
        guard let media = env["VERBATIM_SELF_TEST_MEDIA"],
              let modelDirectory = env["VERBATIM_SELF_TEST_MODEL_DIR"],
              let output = env["VERBATIM_SELF_TEST_OUTPUT"] else { return nil }
        return SelfTest(
            media: URL(fileURLWithPath: media),
            modelDirectory: modelDirectory,
            backend: env["VERBATIM_SELF_TEST_BACKEND"] ?? "faster-whisper",
            output: URL(fileURLWithPath: output, isDirectory: true)
        )
    }

    func run(_ model: AppModel) async {
        var result: [String: Any] = ["backend": backend, "media": media.path]
        do {
            try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
            await model.start()
            guard model.engineState == .ready else { throw Failure(description: "engine not ready: \(model.engineState)") }

            let staged = await model.stage([media])
            if let error = staged.first?.error { throw Failure(description: "validation failed: \(error)") }
            let options = JobOptions(mode: "balanced", backend: backend, modelPath: modelDirectory)
            guard await model.transcribe(staged, options: options) else {
                throw Failure(description: model.alert?.message ?? "could not start the job")
            }
            guard let project = model.projects.first(where: { $0.source.path == media.path }) ?? model.projects.first else {
                throw Failure(description: "project was not created")
            }

            // Completion arrives through the engine's job.updated notifications.
            let deadline = Date().addingTimeInterval(900)
            var job = model.latestJob(for: project.id)
            while Date() < deadline, job?.status.isActive ?? true {
                try await Task.sleep(for: .milliseconds(250))
                job = model.latestJob(for: project.id)
            }
            guard let job, job.status == .completed else {
                throw Failure(description: "job ended as \(job?.status.rawValue ?? "timeout"): \(job?.error?.message ?? "")")
            }
            let transcript = try await model.transcript(for: project)
            result["status"] = "completed"
            result["language"] = transcript.language.map { $0 as Any } ?? NSNull()
            result["segments"] = transcript.segments.count
            result["text"] = transcript.segments.map(\.text).joined(separator: " ")
            result["stages"] = job.stages.map { "\($0.stage):\($0.status.rawValue)" }

            // Show the transcript workspace and snapshot the window.
            model.section = .library
            model.openProjectRequest = project.id
            try await Task.sleep(for: .seconds(4))
            result["snapshot"] = snapshotWindow(to: output.appendingPathComponent("window.png"))
        } catch {
            result["status"] = "failed"
            result["error"] = String(describing: error)
        }
        if let data = try? JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: output.appendingPathComponent("result.json"))
        }
        NSApp.terminate(nil)  // the normal quit path: the engine is stopped cleanly
    }

    private func snapshotWindow(to url: URL) -> Bool {
        guard let window = NSApp.windows.first(where: { $0.isVisible && !($0 is NSPanel) }),
              let view = window.contentView,
              let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { return false }
        view.cacheDisplay(in: view.bounds, to: rep)
        guard let png = rep.representation(using: .png, properties: [:]) else { return false }
        return (try? png.write(to: url)) != nil
    }
}
