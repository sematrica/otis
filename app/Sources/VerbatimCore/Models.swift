import Foundation

// Protocol data types. Field names are the camel-cased form of the engine's
// snake_case keys (decoded with `.convertFromSnakeCase`). Real captured
// examples of every type live in shared/protocol-fixtures/.

// MARK: - Engine

public struct EngineInfo: Codable, Hashable, Sendable {
    public struct Python: Codable, Hashable, Sendable {
        public var version: String
        public var executable: String
    }

    public struct Hardware: Codable, Hashable, Sendable {
        public var platform: String
        public var machine: String
        public var macosVersion: String?
        public var appleSilicon: Bool
        public var performanceCores: Int
        public var logicalCpus: Int?
        public var memoryBytes: Int64?
        public var python: String
    }

    public var version: String
    public var protocolVersion: Int
    public var pid: Int
    public var python: Python
    public var dataDir: String
    public var dbPath: String
    public var schemaVersion: Int
    public var hardware: Hardware
    public var backends: [BackendInfo]
}

public struct BackendCapabilities: Codable, Hashable, Sendable {
    public var beamSearch: Bool
    public var wordTimestamps: Bool
    public var builtinVad: Bool
    public var languageDetection: Bool
    public var device: String
    public var acceleration: String
}

public struct BackendInfo: Codable, Hashable, Sendable, Identifiable {
    public var name: String
    public var displayName: String
    public var available: Bool
    public var unavailableReason: String?
    public var capabilities: BackendCapabilities
    public var id: String { name }
}

// MARK: - Media & projects

public struct AudioStreamInfo: Codable, Hashable, Sendable {
    public var index: Int
    public var codec: String
    public var sampleRate: Int
    public var channels: Int
    public var duration: Double?
    public var language: String?
    public var isDefault: Bool
}

public struct VideoStreamInfo: Codable, Hashable, Sendable {
    public var index: Int
    public var codec: String
    public var width: Int
    public var height: Int
    public var frameRate: Double?
}

public struct MediaInfo: Codable, Hashable, Sendable {
    public var path: String
    public var sizeBytes: Int64
    public var mtimeNs: Int64
    public var formatName: String
    public var formatLongName: String
    public var duration: Double?
    public var startTime: Double
    public var bitRate: Int64?
    public var audioStreams: [AudioStreamInfo]
    public var videoStreams: [VideoStreamInfo]
    public var selectedAudioStream: Int?
    public var hasVideo: Bool
    public var hasAudio: Bool
}

public enum ProjectStatus: String, Codable, Sendable, CaseIterable {
    case new, queued, processing, ready, failed, cancelled, interrupted
}

public struct ProjectSource: Codable, Hashable, Sendable {
    public var kind: String
    public var path: String?
    public var url: String?
    public var sizeBytes: Int64?
    public var mtimeNs: Int64?
}

public struct Project: Codable, Hashable, Sendable, Identifiable {
    public var id: String
    public var title: String
    public var status: ProjectStatus
    public var createdAt: String
    public var updatedAt: String
    public var source: ProjectSource
    public var duration: Double?
    public var hasVideo: Bool
    public var language: String?
    public var media: MediaInfo
    public var lastJobId: String?
    public var segmentCount: Int

    public var sourceURL: URL? { source.path.map { URL(fileURLWithPath: $0) } }
    public var updatedDate: Date? { ISO8601.parse(updatedAt) }
}

// MARK: - Jobs

public enum JobStatus: String, Codable, Sendable {
    case queued, running, completed, failed, cancelled, interrupted

    public var isActive: Bool { self == .queued || self == .running }
    public var isResumable: Bool { self == .failed || self == .cancelled || self == .interrupted }
}

public enum StageStatus: String, Codable, Sendable {
    case pending, running, completed, failed, skipped, cancelled
}

public struct StageProgress: Codable, Hashable, Sendable {
    public var done: Double?
    public var total: Double?
    public var unit: String?
}

public struct JobStage: Codable, Hashable, Sendable, Identifiable {
    public var stage: String
    public var label: String
    public var status: StageStatus
    public var startedAt: String?
    public var finishedAt: String?
    public var elapsedSeconds: Double?
    public var progress: StageProgress?
    /// Present only when the engine has a real numerator and denominator.
    public var fraction: Double?
    /// Present only after enough work has been measured.
    public var etaSeconds: Double?
    public var detail: String?
    public var id: String { stage }
}

public struct ErrorInfo: Codable, Hashable, Sendable {
    public var code: String?
    public var message: String?
}

public struct JobOptions: Codable, Hashable, Sendable {
    public var mode: String
    public var backend: String
    public var modelId: String?
    public var modelPath: String?
    public var language: String?
    public var chunkSeconds: Double?
    public var vad: Bool?

    public init(mode: String = "balanced", backend: String = "auto", modelId: String? = nil, modelPath: String? = nil,
                language: String? = nil, chunkSeconds: Double? = nil, vad: Bool? = nil) {
        self.mode = mode
        self.backend = backend
        self.modelId = modelId
        self.modelPath = modelPath
        self.language = language
        self.chunkSeconds = chunkSeconds
        self.vad = vad
    }
}

public struct Job: Codable, Hashable, Sendable, Identifiable {
    public var id: String
    public var projectId: String
    public var projectTitle: String?
    public var kind: String
    public var status: JobStatus
    public var mode: String
    public var options: JobOptions
    public var effective: JSONValue?
    public var createdAt: String
    public var updatedAt: String
    public var startedAt: String?
    public var finishedAt: String?
    public var elapsedSeconds: Double?
    public var attempt: Int
    public var currentStage: String?
    public var cancelRequested: Bool
    public var error: ErrorInfo?
    public var stages: [JobStage]

    public var runningStage: JobStage? { stages.first { $0.status == .running } }
    public var updatedDate: Date? { ISO8601.parse(updatedAt) }
}

// MARK: - Transcription modes & preflight

public struct TranscriptionMode: Codable, Hashable, Sendable, Identifiable {
    public var id: String
    public var label: String
    public var description: String
    public var family: String
    public var beamSize: Int
    public var bestOf: Int
    public var cpuComputeType: String
    public var limitations: String
}

public struct ModesResult: Codable, Sendable { public var modes: [TranscriptionMode] }

public struct ResolvedModel: Codable, Hashable, Sendable {
    public var id: String?
    public var displayName: String
    public var path: String
    public var revision: String?
}

public struct Preflight: Codable, Hashable, Sendable {
    public struct Resolved: Codable, Hashable, Sendable {
        public var mode: String
        public var backend: String
        public var model: ResolvedModel
        public var chunkSeconds: Double
        public var warnings: [String]
    }

    public var options: JobOptions
    public var resolved: Resolved
}

// MARK: - Transcript

public struct Word: Codable, Hashable, Sendable {
    public var start: Double?
    public var end: Double?
    public var text: String
    public var probability: Double?
    public var timingSource: String?
}

public struct Segment: Codable, Hashable, Sendable, Identifiable {
    public var id: Int
    public var ordinal: Int
    public var start: Double
    public var end: Double
    public var text: String
    public var originalText: String
    public var language: String?
    public var avgLogprob: Double?
    public var noSpeechProb: Double?
    public var compressionRatio: Double?
    public var chunkIndex: Int?
    public var speakerId: Int?
    public var flags: [JSONValue]
    public var words: [Word]?

    /// Flag types recorded by the engine (e.g. "chunk_boundary").
    public var flagTypes: [String] { flags.compactMap { $0["type"]?.stringValue } }
}

public struct Transcript: Codable, Hashable, Sendable {
    public var projectId: String
    public var language: String?
    public var jobId: String?
    public var effective: JSONValue?
    public var segments: [Segment]
}

public struct ExportRendered: Codable, Hashable, Sendable {
    public var content: String
    public var suggestedFilename: String
}

public struct ExportWritten: Codable, Hashable, Sendable {
    public var path: String
    public var bytes: Int
}

// MARK: - Models & settings

public struct ModelStatus: Codable, Hashable, Sendable {
    public var installed: Bool
    public var path: String
    public var sizeOnDisk: Int64
    public var revision: String?
    public var downloadedAt: String?
    public var missing: [String]
}

public struct ModelEntry: Codable, Hashable, Sendable, Identifiable {
    public var id: String
    public var backend: String
    public var family: String
    public var displayName: String
    public var repoId: String
    public var parameters: String
    public var approxDownloadBytes: Int64
    public var memoryEstimates: [String: Int64]
    public var multilingual: Bool
    public var revision: String?
    public var notes: String
    public var status: ModelStatus
}

public struct ModelDownload: Codable, Hashable, Sendable {
    public var modelId: String
    public var status: String
    public var done: Int64?
    public var total: Int64?
    public var totalIsEstimate: Bool?
    public var error: String?

    public var isFinished: Bool { ["completed", "failed", "cancelled"].contains(status) }
}

public struct ModelsListing: Codable, Hashable, Sendable {
    public var modelsDir: String
    public var models: [ModelEntry]
    public var downloads: [String: ModelDownload]
    public var backends: [BackendInfo]
}

public struct EngineSettings: Codable, Hashable, Sendable {
    public var modelsDir: String
    public var defaultModelsDir: String
}

public struct ProjectsResult: Codable, Sendable { public var projects: [Project] }
public struct JobsResult: Codable, Sendable { public var jobs: [Job] }
public struct ProjectDeleted: Codable, Sendable { public var projectId: String }

// MARK: - Dates

public enum ISO8601 {
    public static func parse(_ value: String?) -> Date? {
        guard let value else { return nil }
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = formatter.date(from: value) { return date }
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.date(from: value)
    }
}
