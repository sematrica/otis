import Foundation

// Typed wrappers for every engine method used by the app.

public struct ProjectIDParams: Encodable, Sendable { public let projectId: String }
public struct JobIDParams: Encodable, Sendable { public let jobId: String }
public struct ModelIDParams: Encodable, Sendable { public let modelId: String }
public struct PathParams: Encodable, Sendable { public let path: String }
public struct CreateProjectParams: Encodable, Sendable { public let path: String; public let title: String? }
public struct RenameParams: Encodable, Sendable { public let projectId: String; public let title: String }
public struct StartJobParams: Encodable, Sendable { public let projectId: String; public let options: JobOptions }
public struct PreflightParams: Encodable, Sendable { public let options: JobOptions }
public struct TranscriptParams: Encodable, Sendable { public let projectId: String; public let includeWords: Bool }
public struct ListJobsParams: Encodable, Sendable { public let activeOnly: Bool; public let limit: Int }
public struct ModelsDirParams: Encodable, Sendable {
    public let modelsDir: String?
    // Encode an explicit null so the engine resets to the default location.
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelsDir, forKey: .modelsDir)
    }
    enum CodingKeys: String, CodingKey { case modelsDir }
}

public struct ExportOptions: Codable, Hashable, Sendable {
    public var timestamps: Bool
    public var header: Bool
    public init(timestamps: Bool = true, header: Bool = true) {
        self.timestamps = timestamps
        self.header = header
    }
}

public struct ExportParams: Encodable, Sendable {
    public let projectId: String
    public let format: String
    public let path: String?
    public let options: ExportOptions
}

public extension EngineClient {
    func info() async throws -> EngineInfo { try await call("engine.info", as: EngineInfo.self) }

    func probe(path: String) async throws -> MediaInfo {
        try await call("media.probe", params: PathParams(path: path), as: MediaInfo.self)
    }

    func createProject(path: String, title: String? = nil) async throws -> Project {
        try await call("projects.create", params: CreateProjectParams(path: path, title: title), as: Project.self)
    }

    func listProjects() async throws -> [Project] {
        try await call("projects.list", as: ProjectsResult.self).projects
    }

    func project(_ id: String) async throws -> Project {
        try await call("projects.get", params: ProjectIDParams(projectId: id), as: Project.self)
    }

    func renameProject(_ id: String, title: String) async throws -> Project {
        try await call("projects.rename", params: RenameParams(projectId: id, title: title), as: Project.self)
    }

    func deleteProject(_ id: String) async throws {
        _ = try await call("projects.delete", params: ProjectIDParams(projectId: id), as: JSONValue.self, timeout: .seconds(60))
    }

    func modes() async throws -> [TranscriptionMode] {
        try await call("transcription.modes", as: ModesResult.self).modes
    }

    func preflight(_ options: JobOptions) async throws -> Preflight {
        try await call("transcription.preflight", params: PreflightParams(options: options), as: Preflight.self)
    }

    func startJob(projectId: String, options: JobOptions) async throws -> Job {
        try await call("jobs.start", params: StartJobParams(projectId: projectId, options: options), as: Job.self)
    }

    func cancelJob(_ id: String) async throws -> Job {
        try await call("jobs.cancel", params: JobIDParams(jobId: id), as: Job.self)
    }

    func resumeJob(_ id: String) async throws -> Job {
        try await call("jobs.resume", params: JobIDParams(jobId: id), as: Job.self)
    }

    func job(_ id: String) async throws -> Job {
        try await call("jobs.get", params: JobIDParams(jobId: id), as: Job.self)
    }

    func listJobs(activeOnly: Bool = false, limit: Int = 200) async throws -> [Job] {
        try await call("jobs.list", params: ListJobsParams(activeOnly: activeOnly, limit: limit), as: JobsResult.self).jobs
    }

    func transcript(projectId: String, includeWords: Bool = false) async throws -> Transcript {
        try await call("transcript.get", params: TranscriptParams(projectId: projectId, includeWords: includeWords),
                       as: Transcript.self, timeout: .seconds(120))
    }

    func renderExport(projectId: String, format: String = "txt", options: ExportOptions = .init()) async throws -> ExportRendered {
        try await call("export.render", params: ExportParams(projectId: projectId, format: format, path: nil, options: options),
                       as: ExportRendered.self)
    }

    func writeExport(projectId: String, to path: String, format: String = "txt", options: ExportOptions = .init()) async throws -> ExportWritten {
        try await call("export.write", params: ExportParams(projectId: projectId, format: format, path: path, options: options),
                       as: ExportWritten.self)
    }

    func models() async throws -> ModelsListing { try await call("models.list", as: ModelsListing.self) }

    func downloadModel(_ id: String) async throws -> ModelDownload {
        try await call("models.download", params: ModelIDParams(modelId: id), as: ModelDownload.self)
    }

    func cancelDownload(_ id: String) async throws {
        _ = try await call("models.cancelDownload", params: ModelIDParams(modelId: id), as: JSONValue.self)
    }

    func deleteModel(_ id: String) async throws {
        _ = try await call("models.delete", params: ModelIDParams(modelId: id), as: JSONValue.self)
    }

    func settings() async throws -> EngineSettings { try await call("settings.get", as: EngineSettings.self) }

    func setModelsDirectory(_ path: String?) async throws -> EngineSettings {
        try await call("settings.set", params: ModelsDirParams(modelsDir: path), as: EngineSettings.self)
    }
}
