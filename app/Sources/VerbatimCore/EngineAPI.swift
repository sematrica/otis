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

public struct RenameSpeakerParams: Encodable, Sendable {
    public let projectId: String
    public let speakerId: Int
    public let name: String?
    // Encode an explicit null: clearing the name restores the speaker's label.
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(projectId, forKey: .projectId)
        try container.encode(speakerId, forKey: .speakerId)
        try container.encode(name, forKey: .name)
    }
    enum CodingKeys: String, CodingKey { case projectId, speakerId, name }
}
public struct CreateSpeakerParams: Encodable, Sendable { public let projectId: String; public let name: String? }
public struct MergeSpeakersParams: Encodable, Sendable { public let projectId: String; public let speakerIds: [Int]; public let into: Int }
public struct AssignSpeakerParams: Encodable, Sendable {
    public let projectId: String
    public let segmentIds: [Int]
    public let speakerId: Int?
    public let newSpeakerName: String?
}
public struct SplitSegmentParams: Encodable, Sendable { public let projectId: String; public let segmentId: Int; public let wordIndex: Int }
public struct MergeSegmentsParams: Encodable, Sendable { public let projectId: String; public let segmentIds: [Int] }
public struct EditTextParams: Encodable, Sendable { public let projectId: String; public let segmentId: Int; public let text: String }
public struct MarkUncertainParams: Encodable, Sendable {
    public let projectId: String
    public let segmentId: Int
    public let uncertain: Bool
    public let note: String?
}

public struct SegmentParams: Encodable, Sendable { public let projectId: String; public let segmentId: Int }
public struct ChooseWordingParams: Encodable, Sendable { public let projectId: String; public let segmentId: Int; public let source: String }
public struct MarkReviewedParams: Encodable, Sendable { public let projectId: String; public let segmentIds: [Int]; public let reviewed: Bool }
public struct SegmentsParams: Encodable, Sendable { public let projectId: String; public let segmentIds: [Int] }
public struct LLMProviderParams: Encodable, Sendable {
    public let provider: String
    public let model: String?
    public let baseUrl: String?
    public let apiKey: String?
}
public struct LLMReviewParams: Encodable, Sendable {
    public let projectId: String
    public let provider: String
    public let model: String
    public let baseUrl: String?
    public let apiKey: String?
    public let allowRemote: Bool
}
public struct RunIDParams: Encodable, Sendable { public let runId: Int; public let exchanges: Bool }
public struct SuggestionIDParams: Encodable, Sendable { public let suggestionId: Int }
public struct SuggestionsParams: Encodable, Sendable { public let projectId: String; public let statuses: [String]? }

public struct ExportOptions: Codable, Hashable, Sendable {
    public var timestamps: Bool
    public var header: Bool
    /// CSV: one row per word instead of per segment.
    public var words: Bool?
    public init(timestamps: Bool = true, header: Bool = true, words: Bool? = nil) {
        self.timestamps = timestamps
        self.header = header
        self.words = words
    }
}

/// The engine's export formats (``export.formats``).
public enum ExportFormat: String, CaseIterable, Sendable, Identifiable {
    case txt, srt, vtt, json, csv, md

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .txt: return "Plain Text"
        case .srt: return "SubRip Subtitles (SRT)"
        case .vtt: return "WebVTT Subtitles"
        case .json: return "JSON (everything, with words and the original transcript)"
        case .csv: return "CSV Spreadsheet"
        case .md: return "Markdown"
        }
    }

    public var fileExtension: String { rawValue }
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

    func originalTranscript(projectId: String) async throws -> OriginalTranscript {
        try await call("transcript.original", params: ProjectIDParams(projectId: projectId), as: OriginalTranscript.self,
                       timeout: .seconds(120))
    }

    func editHistory(projectId: String) async throws -> [EditRecord] {
        try await call("transcript.history", params: ProjectIDParams(projectId: projectId), as: EditHistoryResult.self).edits
    }

    // Corrections. Each returns the updated transcript (with words).

    func renameSpeaker(projectId: String, speakerId: Int, name: String?) async throws -> EditResult {
        try await edit("transcript.renameSpeaker", RenameSpeakerParams(projectId: projectId, speakerId: speakerId, name: name))
    }

    func createSpeaker(projectId: String, name: String?) async throws -> EditResult {
        try await edit("transcript.createSpeaker", CreateSpeakerParams(projectId: projectId, name: name))
    }

    func mergeSpeakers(projectId: String, speakerIds: [Int], into: Int) async throws -> EditResult {
        try await edit("transcript.mergeSpeakers", MergeSpeakersParams(projectId: projectId, speakerIds: speakerIds, into: into))
    }

    func assignSpeaker(projectId: String, segmentIds: [Int], speakerId: Int) async throws -> EditResult {
        try await edit("transcript.assignSpeaker",
                       AssignSpeakerParams(projectId: projectId, segmentIds: segmentIds, speakerId: speakerId, newSpeakerName: nil))
    }

    func assignNewSpeaker(projectId: String, segmentIds: [Int], name: String) async throws -> EditResult {
        try await edit("transcript.assignSpeaker",
                       AssignSpeakerParams(projectId: projectId, segmentIds: segmentIds, speakerId: nil, newSpeakerName: name))
    }

    func splitSegment(projectId: String, segmentId: Int, wordIndex: Int) async throws -> EditResult {
        try await edit("transcript.splitSegment", SplitSegmentParams(projectId: projectId, segmentId: segmentId, wordIndex: wordIndex))
    }

    func mergeSegments(projectId: String, segmentIds: [Int]) async throws -> EditResult {
        try await edit("transcript.mergeSegments", MergeSegmentsParams(projectId: projectId, segmentIds: segmentIds))
    }

    func editText(projectId: String, segmentId: Int, text: String) async throws -> EditResult {
        try await edit("transcript.editText", EditTextParams(projectId: projectId, segmentId: segmentId, text: text))
    }

    func markUncertain(projectId: String, segmentId: Int, uncertain: Bool, note: String? = nil) async throws -> EditResult {
        try await edit("transcript.markUncertain",
                       MarkUncertainParams(projectId: projectId, segmentId: segmentId, uncertain: uncertain, note: note))
    }

    func undo(projectId: String) async throws -> EditResult {
        try await edit("transcript.undo", ProjectIDParams(projectId: projectId))
    }

    func redo(projectId: String) async throws -> EditResult {
        try await edit("transcript.redo", ProjectIDParams(projectId: projectId))
    }

    // MARK: Stage 3: review and re-check

    func deleteSegment(projectId: String, segmentId: Int) async throws -> EditResult {
        try await edit("transcript.deleteSegment", SegmentParams(projectId: projectId, segmentId: segmentId))
    }

    /// Use a wording recorded by re-checking ("first_pass" or a candidate's source).
    func chooseWording(projectId: String, segmentId: Int, source: String) async throws -> EditResult {
        try await edit("transcript.chooseWording", ChooseWordingParams(projectId: projectId, segmentId: segmentId, source: source))
    }

    func markReviewed(projectId: String, segmentIds: [Int], reviewed: Bool) async throws -> EditResult {
        try await edit("transcript.markReviewed", MarkReviewedParams(projectId: projectId, segmentIds: segmentIds, reviewed: reviewed))
    }

    /// Re-transcribe passages and choose their wording by acoustic evidence (a job).
    func recheck(projectId: String, segmentIds: [Int]) async throws -> Job {
        try await call("transcript.recheck", params: SegmentsParams(projectId: projectId, segmentIds: segmentIds), as: Job.self)
    }

    // MARK: Stage 3: AI verification

    func llmModels(provider: String, baseURL: String?, apiKey: String?) async throws -> LLMModelsResult {
        try await call("llm.models", params: LLMProviderParams(provider: provider, model: nil, baseUrl: baseURL, apiKey: apiKey),
                       as: LLMModelsResult.self, timeout: .seconds(60))
    }

    func llmReview(projectId: String, provider: String, model: String, baseURL: String?, apiKey: String?,
                   allowRemote: Bool) async throws -> LLMRun {
        try await call("llm.review", params: LLMReviewParams(projectId: projectId, provider: provider, model: model,
                                                             baseUrl: baseURL, apiKey: apiKey, allowRemote: allowRemote),
                       as: LLMRun.self)
    }

    func llmRuns(projectId: String) async throws -> [LLMRun] {
        try await call("llm.runs", params: ProjectIDParams(projectId: projectId), as: LLMRunsResult.self).runs
    }

    func llmRun(id: Int, exchanges: Bool = false) async throws -> LLMRun {
        try await call("llm.run", params: RunIDParams(runId: id, exchanges: exchanges), as: LLMRun.self)
    }

    func cancelLLMRun(id: Int) async throws -> LLMRun {
        try await call("llm.cancel", params: RunIDParams(runId: id, exchanges: false), as: LLMRun.self)
    }

    func suggestions(projectId: String, statuses: [String]? = nil) async throws -> [Suggestion] {
        try await call("suggestions.list", params: SuggestionsParams(projectId: projectId, statuses: statuses),
                       as: SuggestionsResult.self).suggestions
    }

    func acceptSuggestion(id: Int) async throws -> SuggestionAcceptResult {
        try await call("suggestions.accept", params: SuggestionIDParams(suggestionId: id), as: SuggestionAcceptResult.self,
                       timeout: .seconds(120))
    }

    func rejectSuggestion(id: Int) async throws -> Suggestion {
        try await call("suggestions.reject", params: SuggestionIDParams(suggestionId: id), as: Suggestion.self)
    }

    private func edit<Params: Encodable & Sendable>(_ method: String, _ params: Params) async throws -> EditResult {
        try await call(method, params: params, as: EditResult.self, timeout: .seconds(120))
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
