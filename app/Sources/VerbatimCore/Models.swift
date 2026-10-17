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

/// An option that is on, off, or decided by the engine ("auto": used when the
/// model it needs is installed). Encoded as `true`, `false` or `"auto"`.
public enum Tristate: Hashable, Sendable, Codable {
    case on, off, auto

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(Bool.self) {
            self = value ? .on : .off
        } else {
            let text = try container.decode(String.self)
            guard text == "auto" else {
                throw DecodingError.dataCorruptedError(in: container, debugDescription: "unexpected value \(text)")
            }
            self = .auto
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .on: try container.encode(true)
        case .off: try container.encode(false)
        case .auto: try container.encode("auto")
        }
    }
}

/// What the user knows about the number of speakers (all nil: estimate).
public struct SpeakerBounds: Codable, Hashable, Sendable {
    public var num: Int?
    public var min: Int?
    public var max: Int?

    public init(num: Int? = nil, min: Int? = nil, max: Int? = nil) {
        self.num = num
        self.min = min
        self.max = max
    }
}

public struct JobOptions: Codable, Hashable, Sendable {
    public var mode: String
    public var backend: String
    public var modelId: String?
    public var modelPath: String?
    public var language: String?
    public var chunkSeconds: Double?
    public var vad: Bool?
    public var diarize: Tristate?
    public var speakers: SpeakerBounds?
    public var align: Tristate?
    public var alignmentModel: String?
    /// Re-transcribe doubtful passages and choose by acoustic evidence
    /// ("auto": on in Maximum Accuracy mode).
    public var recheck: Tristate?

    public init(mode: String = "balanced", backend: String = "auto", modelId: String? = nil, modelPath: String? = nil,
                language: String? = nil, chunkSeconds: Double? = nil, vad: Bool? = nil, diarize: Tristate? = nil,
                speakers: SpeakerBounds? = nil, align: Tristate? = nil, alignmentModel: String? = nil,
                recheck: Tristate? = nil) {
        self.mode = mode
        self.backend = backend
        self.modelId = modelId
        self.modelPath = modelPath
        self.language = language
        self.chunkSeconds = chunkSeconds
        self.vad = vad
        self.diarize = diarize
        self.speakers = speakers
        self.align = align
        self.alignmentModel = alignmentModel
        self.recheck = recheck
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
    /// Place in the processing queue (1 = next), while queued.
    public var queuePosition: Int?

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
    /// Forced word alignment by default in this mode.
    public var align: Bool?
    /// Doubtful passages are re-transcribed and compared against the audio.
    public var recheck: Bool?
}

public struct ModesResult: Codable, Sendable { public var modes: [TranscriptionMode] }

public struct ResolvedModel: Codable, Hashable, Sendable {
    public var id: String?
    public var displayName: String
    public var path: String
    public var revision: String?
}

public struct Preflight: Codable, Hashable, Sendable {
    public struct Diarization: Codable, Hashable, Sendable {
        public var enabled: Bool
        public var skipReason: String?
    }

    public struct Alignment: Codable, Hashable, Sendable {
        public var enabled: Bool
        public var candidates: [String]
        public var skipReason: String?
    }

    /// Re-transcription of doubtful passages (Maximum Accuracy), and the
    /// installed models that can compare the wordings by acoustic evidence.
    public struct Recheck: Codable, Hashable, Sendable {
        public var enabled: Bool
        public var skipReason: String?
        public var acousticModels: [String]
    }

    public struct Resolved: Codable, Hashable, Sendable {
        public var mode: String
        public var backend: String
        public var model: ResolvedModel
        public var chunkSeconds: Double
        public var warnings: [String]
        public var diarization: Diarization?
        public var alignment: Alignment?
        public var recheck: Recheck?
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
    /// "whisper", "alignment", "wildcard" or "interpolated".
    public var timingSource: String?
    public var speakerId: Int?
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
    /// The diarization's speaker for this segment (kept when the user changes it).
    public var originalSpeakerId: Int?
    public var speakerConfidence: Double?
    public var textEdited: Bool?
    public var speakerEdited: Bool?
    public var flags: [JSONValue]
    public var words: [Word]?
    /// "high", "medium" or "low": the engine's review priority (most severe open
    /// flag); nil when nothing asks for a look or the user marked it reviewed.
    public var reviewPriority: String?

    /// Flag types recorded by the engine (e.g. "chunk_boundary").
    public var flagTypes: [String] { flags.compactMap { $0["type"]?.stringValue } }

    public func flag(_ type: String) -> JSONValue? {
        flags.first { $0["type"]?.stringValue == type }
    }

    public var isTextEdited: Bool { textEdited ?? false }
    public var isSpeakerEdited: Bool { speakerEdited ?? false }
    /// Something here asks for a look (high or medium priority).
    public var needsReview: Bool { reviewPriority == "high" || reviewPriority == "medium" }
    /// The user marked this passage as checked.
    public var isReviewed: Bool { flagTypes.contains("reviewed") }
}

public struct Speaker: Codable, Hashable, Sendable, Identifiable {
    public var id: Int
    /// "Speaker 1", "Speaker 2", ... in order of first appearance.
    public var label: String
    /// A name the user gave; never set automatically.
    public var name: String?
    public var displayName: String
    /// "diarization" or "user".
    public var origin: String
    public var colorIndex: Int
    public var mergedInto: Int?
    public var segmentCount: Int
    public var speakingSeconds: Double
    public var firstStart: Double?
}

public struct EditHistory: Codable, Hashable, Sendable {
    public var canUndo: Bool
    public var canRedo: Bool
    public var undoSummary: String?
    public var redoSummary: String?
    public var editCount: Int

    public static let empty = EditHistory(canUndo: false, canRedo: false, undoSummary: nil, redoSummary: nil, editCount: 0)
}

public struct Transcript: Codable, Hashable, Sendable {
    public var projectId: String
    public var language: String?
    public var jobId: String?
    public var effective: JSONValue?
    public var revision: Int
    public var history: EditHistory
    public var speakers: [Speaker]
    public var mergedSpeakers: [Speaker]
    public var segments: [Segment]

    enum CodingKeys: String, CodingKey {
        case projectId, language, jobId, effective, revision, history, speakers, mergedSpeakers, segments
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        projectId = try c.decode(String.self, forKey: .projectId)
        language = try c.decodeIfPresent(String.self, forKey: .language)
        jobId = try c.decodeIfPresent(String.self, forKey: .jobId)
        effective = try c.decodeIfPresent(JSONValue.self, forKey: .effective)
        revision = try c.decodeIfPresent(Int.self, forKey: .revision) ?? 0
        history = try c.decodeIfPresent(EditHistory.self, forKey: .history) ?? .empty
        speakers = try c.decodeIfPresent([Speaker].self, forKey: .speakers) ?? []
        mergedSpeakers = try c.decodeIfPresent([Speaker].self, forKey: .mergedSpeakers) ?? []
        segments = try c.decode([Segment].self, forKey: .segments)
    }

    /// Start times of detected speech that no word covers, as found by the
    /// final validation of the job that produced this transcript (listed up to
    /// a limit; ``untranscribedSpeechCount`` counts them all).
    public var untranscribedSpeech: [Double] {
        guard case .array(let items)? = effective?.lookup("validation")?.lookup("untranscribed") else { return [] }
        return items.compactMap { $0.lookup("start")?.doubleValue }
    }

    public var untranscribedSpeechCount: Int {
        Int(effective?.lookup("validation")?.lookup("missing_speech")?.doubleValue ?? 0)
    }

    /// Any speaker by id, including ones the user merged away.
    public func speaker(_ id: Int?) -> Speaker? {
        guard let id else { return nil }
        return speakers.first { $0.id == id } ?? mergedSpeakers.first { $0.id == id }
    }
}

/// The AI-generated transcript as first stored; corrections never change it.
public struct OriginalTranscript: Codable, Hashable, Sendable {
    public var projectId: String
    public var jobId: String?
    public var createdAt: String
    public var language: String?
    public var speakers: [Speaker]
    public var segments: [Segment]
}

/// Result of a correction, undo or redo.
public struct EditResult: Codable, Hashable, Sendable {
    public var summary: String?
    public var changed: Bool
    public var transcript: Transcript
}

public struct EditRecord: Codable, Hashable, Sendable, Identifiable {
    public var id: Int
    public var kind: String
    public var summary: String
    public var undone: Bool
    public var createdAt: String
}

public struct EditHistoryResult: Codable, Sendable { public var edits: [EditRecord] }

/// The processing queue (``queue.get``), persisted by the engine.
public struct QueueState: Codable, Sendable {
    public var paused: Bool
    /// Interrupted work is queued again when the engine starts.
    public var autoResume: Bool
    public var running: Job?
    /// In the order they will run.
    public var queued: [Job]
}

/// ``queue.updated``: the queue's order or paused state changed.
public struct QueueUpdate: Codable, Hashable, Sendable {
    public var paused: Bool
    public var autoResume: Bool
    public var running: String?
    public var queued: [String]

    public init(paused: Bool, autoResume: Bool, running: String?, queued: [String]) {
        self.paused = paused
        self.autoResume = autoResume
        self.running = running
        self.queued = queued
    }
}

public struct TranscriptUpdated: Codable, Hashable, Sendable {
    public var projectId: String
    public var revision: Int
    public var summary: String?
}

// MARK: - Search

/// A passage found by ``search.transcripts``. ``highlights`` are ranges of
/// ``text`` in Unicode scalars (the engine's string offsets).
public struct SearchHit: Codable, Hashable, Sendable, Identifiable {
    public var projectId: String
    public var projectTitle: String
    public var segmentId: Int
    public var start: Double
    public var end: Double
    public var speakerId: Int?
    public var speaker: String?
    public var text: String
    public var highlights: [[Int]]
    /// The match is only in the AI's wording, which the user corrected.
    public var inOriginalOnly: Bool

    public var id: Int { segmentId }

    /// ``text`` split into (piece, highlighted) runs.
    public var runs: [(String, Bool)] {
        let scalars = Array(text.unicodeScalars)
        var out: [(String, Bool)] = []
        var position = 0
        func piece(_ a: Int, _ b: Int) -> String {
            var view = String.UnicodeScalarView()
            view.append(contentsOf: scalars[max(0, a)..<min(scalars.count, b)])
            return String(view)
        }
        for range in highlights where range.count == 2 && range[0] >= position && range[0] < range[1]
            && range[0] < scalars.count {
            if range[0] > position { out.append((piece(position, range[0]), false)) }
            out.append((piece(range[0], range[1]), true))
            position = min(range[1], scalars.count)
        }
        if position < scalars.count { out.append((piece(position, scalars.count), false)) }
        return out
    }
}

public struct SearchResponse: Codable, Sendable {
    public var query: String
    public var terms: [String]
    /// "index" (FTS5 words) or "substring" (languages without spaces).
    public var method: String
    public var results: [SearchHit]
    public var truncated: Bool
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
    /// HuggingFace repository, or nil for models downloaded from fixed URLs.
    public var repoId: String?
    /// "asr", "diarization" or "alignment".
    public var kind: String?
    public var license: String?
    /// Supported language codes; empty means any language.
    public var languages: [String]?
    public var homepage: String?
    public var files: [ModelFile]?
    public var parameters: String
    public var approxDownloadBytes: Int64
    public var memoryEstimates: [String: Int64]
    public var multilingual: Bool
    public var revision: String?
    public var notes: String
    public var status: ModelStatus
}

public struct ModelFile: Codable, Hashable, Sendable {
    public var url: String
    public var size: Int64
    public var sha256: String
}

public extension ModelEntry {
    var modelKind: String { kind ?? "asr" }

    /// Where the files come from, for confirmation dialogs.
    var sourceDescription: String {
        if let repoId { return "huggingface.co/\(repoId)" }
        if let host = files?.first.flatMap({ URL(string: $0.url)?.host }) { return host }
        return "the model's publisher"
    }
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

// MARK: - AI verification (Stage 3)

/// One AI review of a transcript: which provider and model, where it ran,
/// and what came of it. Only transcript text is ever sent, never audio.
public struct LLMRun: Codable, Hashable, Sendable, Identifiable {
    public var id: Int
    public var projectId: String
    public var provider: String
    public var model: String
    public var endpoint: String
    /// The endpoint is not on this computer.
    public var remote: Bool
    /// "running", "completed", "failed" or "cancelled".
    public var status: String
    public var error: String?
    public var transcriptRevision: Int
    public var segmentsSent: Int
    public var requests: Int
    public var startedAt: String
    public var finishedAt: String?
    /// Suggestion counts by status.
    public var suggestions: [String: Int]?
    /// Every request and response (only when asked for).
    public var exchanges: [LLMExchange]?

    public var isRunning: Bool { status == "running" }
}

public struct LLMExchange: Codable, Hashable, Sendable {
    public var request: JSONValue
    public var response: String?
    public var seconds: Double
    public var error: String?
}

public struct LLMRunsResult: Codable, Sendable { public var runs: [LLMRun] }

public struct LLMModelsResult: Codable, Sendable {
    public var provider: String
    public var endpoint: String
    public var remote: Bool
    public var models: [String]
}

/// A suggestion from an AI review. Never applied until the user accepts it.
public struct Suggestion: Codable, Hashable, Sendable, Identifiable {
    public var id: Int
    public var projectId: String
    public var runId: Int
    public var segmentId: Int?
    /// "suspicious", "punctuation", "terminology" or "retranscribe".
    public var kind: String
    public var originalText: String
    public var suggestedText: String?
    public var reason: String?
    /// "pending", "accepted", "rejected", "invalid" (failed validation) or "outdated".
    public var status: String
    /// Why validation rejected it.
    public var validation: String?
    public var createdAt: String
    public var decidedAt: String?
}

public struct SuggestionsResult: Codable, Sendable { public var suggestions: [Suggestion] }

public struct SuggestionAcceptResult: Codable, Sendable {
    public var suggestion: Suggestion
    public var summary: String?
    public var changed: Bool
    public var transcript: Transcript?
    /// The acoustic re-check started for a "re-transcribe" suggestion.
    public var job: Job?
}
