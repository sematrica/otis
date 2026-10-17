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
///     VERBATIM_SELF_TEST_SPEAKERS    "1": Maximum Accuracy with required
///                                    speaker identification and word
///                                    alignment; then rename a speaker, undo
///                                    it, and re-check two passages
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
    let speakers: Bool

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
            output: URL(fileURLWithPath: output, isDirectory: true),
            speakers: env["VERBATIM_SELF_TEST_SPEAKERS"] == "1"
        )
    }

    /// Appends a line to progress.log so a hang can be located from outside.
    private func step(_ message: String) {
        let line = "\(Date().timeIntervalSince1970) \(message)\n"
        let url = output.appendingPathComponent("progress.log")
        if let handle = try? FileHandle(forWritingTo: url) {
            handle.seekToEndOfFile()
            handle.write(Data(line.utf8))
            try? handle.close()
        } else {
            try? Data(line.utf8).write(to: url)
        }
    }

    func run(_ model: AppModel) async {
        var result: [String: Any] = ["backend": backend, "media": media.path]
        do {
            try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
            step("starting engine")
            await model.start()
            step("engine state \(model.engineState)")
            guard model.engineState == .ready else { throw Failure(description: "engine not ready: \(model.engineState)") }

            let staged = await model.stage([media])
            step("staged: \(staged.first?.error ?? "ok")")
            if let error = staged.first?.error { throw Failure(description: "validation failed: \(error)") }
            let options = JobOptions(mode: speakers ? "maximum" : "balanced", backend: backend, modelPath: modelDirectory,
                                     diarize: speakers ? .on : .auto, align: speakers ? .on : .auto)
            guard await model.transcribe(staged, options: options) else {
                throw Failure(description: model.alert?.message ?? "could not start the job")
            }
            guard let project = model.projects.first(where: { $0.source.path == media.path }) ?? model.projects.first else {
                throw Failure(description: "project was not created")
            }
            step("job queued for project \(project.id)")

            let job = try await finished(model, project: project.id)
            step("job \(job.status.rawValue)")
            let transcript = try await model.transcript(for: project)
            step("transcript loaded: \(transcript.segments.count) segments")
            result["status"] = "completed"
            result["language"] = transcript.language.map { $0 as Any } ?? NSNull()
            result["segments"] = transcript.segments.count
            result["text"] = transcript.segments.map(\.text).joined(separator: " ")
            result["stages"] = job.stages.map { "\($0.stage):\($0.status.rawValue)" }
            result["speakers"] = transcript.speakers.map(\.label)
            let words = transcript.segments.flatMap { $0.words ?? [] }
            result["words"] = words.count
            result["aligned_words"] = words.filter { $0.timingSource == "alignment" }.count

            if speakers, let first = transcript.speakers.first {
                // A correction through the same client the workspace uses.
                let renamed = try await model.client.renameSpeaker(projectId: project.id, speakerId: first.id, name: "Diane")
                result["renamed"] = renamed.transcript.speaker(first.id).map { $0.displayName as Any } ?? NSNull()
                step("renamed \(first.label)")
            }

            // Show the transcript workspace and snapshot the window.
            model.section = .library
            model.openProjectRequest = project.id
            try await Task.sleep(for: .seconds(4))
            step("workspace opened; taking snapshot")
            result["snapshot"] = snapshotWindow(to: output.appendingPathComponent("window.png"))
            step("snapshot done")

            if speakers, let first = transcript.speakers.first {
                let undone = try await model.client.undo(projectId: project.id)
                result["after_undo"] = undone.transcript.speaker(first.id).map { $0.displayName as Any } ?? NSNull()
                result["can_redo"] = undone.transcript.history.canRedo
                step("undo done")

                // Re-check two passages on demand: a job of its own, saved as one correction.
                let ids = undone.transcript.segments.filter { !($0.words ?? []).isEmpty }.prefix(2).map(\.id)
                let queued = try await model.client.recheck(projectId: project.id, segmentIds: Array(ids))
                step("re-check queued: \(queued.id)")
                let recheck = try await finished(model, project: project.id, after: queued.id)
                let rechecked = try await model.transcript(for: project)
                result["recheck_stages"] = recheck.stages.map { "\($0.stage):\($0.status.rawValue)" }
                result["rechecked"] = rechecked.segments.filter { ids.contains($0.id) && $0.flagTypes.contains("rechecked") }.count
                result["recheck_summary"] = rechecked.history.undoSummary.map { $0 as Any } ?? NSNull()
                step("re-check \(recheck.status.rawValue)")
            }
        } catch {
            result["status"] = "failed"
            result["error"] = String(describing: error)
        }
        if let data = try? JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: output.appendingPathComponent("result.json"))
        }
        step("result written; quitting")
        NSApp.terminate(nil)  // the normal quit path: the engine is stopped cleanly
    }

    /// Waits for the project's latest job (or exactly ``after``) to finish;
    /// completion arrives through the engine's job.updated notifications.
    private func finished(_ model: AppModel, project: String, after id: String? = nil) async throws -> Job {
        let deadline = Date().addingTimeInterval(900)
        var job = id.flatMap { model.jobs[$0] } ?? model.latestJob(for: project)
        while Date() < deadline, job?.status.isActive ?? true {
            try await Task.sleep(for: .milliseconds(250))
            job = id.flatMap { model.jobs[$0] } ?? model.latestJob(for: project)
        }
        guard let job, job.status == .completed else {
            throw Failure(description: "job ended as \(job?.status.rawValue ?? "timeout"): \(job?.error?.message ?? "")")
        }
        return job
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
