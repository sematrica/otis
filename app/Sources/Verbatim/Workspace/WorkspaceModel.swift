import Foundation
import VerbatimCore

/// State of one open transcript workspace: the transcript with words and
/// speakers, the selection, and every correction (each one undoable; the
/// engine keeps the history, so undo survives quitting the app).
@MainActor
final class WorkspaceModel: ObservableObject {
    enum Filter: String, CaseIterable, Identifiable {
        case all, speaker, review
        var id: String { rawValue }
    }

    let projectID: String
    @Published private(set) var transcript: Transcript?
    @Published private(set) var loadError: String?
    /// A correction is being applied (further edits wait for it).
    @Published private(set) var isWorking = false
    @Published var selectedSegmentID: Int?
    @Published var focusedSpeakerID: Int?
    @Published var filter: Filter = .all
    /// Summary of the last change, shown briefly in the transcript header.
    @Published private(set) var lastChange: String?
    private(set) var index = TranscriptIndex(segments: [])

    private let app: AppModel

    init(projectID: String, app: AppModel) {
        self.projectID = projectID
        self.app = app
    }

    private var client: EngineClient { app.client }

    // MARK: Loading

    /// Loads the transcript unless the one shown is already at `revision`.
    func refresh(knownRevision revision: Int?) async {
        if let revision, let transcript, transcript.revision >= revision { return }
        await load()
    }

    func load() async {
        do {
            apply(try await client.transcript(projectId: projectID, includeWords: true))
            loadError = nil
        } catch {
            loadError = error.localizedDescription
        }
    }

    private func apply(_ transcript: Transcript) {
        self.transcript = transcript
        index = TranscriptIndex(segments: transcript.segments)
        if let selected = selectedSegmentID, !transcript.segments.contains(where: { $0.id == selected }) {
            selectedSegmentID = nil
        }
        if let focused = focusedSpeakerID, !transcript.speakers.contains(where: { $0.id == focused }) {
            focusedSpeakerID = nil
            if filter == .speaker { filter = .all }
        }
    }

    // MARK: Derived

    var speakers: [Speaker] { transcript?.speakers ?? [] }
    var segments: [Segment] { transcript?.segments ?? [] }
    var history: EditHistory { transcript?.history ?? .empty }

    func speaker(_ id: Int?) -> Speaker? { transcript?.speaker(id) }

    func segment(_ id: Int?) -> Segment? {
        guard let id else { return nil }
        return segments.first { $0.id == id }
    }

    /// The segment being played at `time`.
    func activeSegmentID(at time: Double) -> Int? {
        guard let i = index.segmentIndex(at: time), i < segments.count else { return nil }
        return segments[i].id
    }

    var visibleSegments: [Segment] {
        switch filter {
        case .all:
            return segments
        case .speaker:
            guard let focusedSpeakerID else { return segments }
            return segments.filter { $0.speakerId == focusedSpeakerID }
        case .review:
            return segments.filter { Self.needsReview($0) }
        }
    }

    static let reviewFlags: Set<String> = [
        "speaker_uncertain", "speaker_change_uncertain", "overlapping_speech", "speaker_unassigned",
        "alignment_failed", "marked_uncertain",
    ]

    static func needsReview(_ segment: Segment) -> Bool {
        !reviewFlags.isDisjoint(with: segment.flagTypes)
    }

    var reviewCount: Int { segments.filter(Self.needsReview).count }

    func neighbour(of segment: Segment, offset: Int) -> Segment? {
        guard let i = segments.firstIndex(where: { $0.id == segment.id }) else { return nil }
        let j = i + offset
        return segments.indices.contains(j) ? segments[j] : nil
    }

    // MARK: Corrections

    private func perform(_ failureTitle: String, _ operation: @Sendable (EngineClient) async throws -> EditResult) async {
        guard !isWorking else { return }
        isWorking = true
        defer { isWorking = false }
        do {
            let result = try await operation(client)
            apply(result.transcript)
            if let summary = result.summary { lastChange = summary }
        } catch {
            app.report(error, title: failureTitle)
        }
    }

    func rename(_ speaker: Speaker, to name: String) async {
        let pid = projectID
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        await perform("Could not rename the speaker") {
            try await $0.renameSpeaker(projectId: pid, speakerId: speaker.id, name: trimmed.isEmpty ? nil : trimmed)
        }
    }

    func merge(_ source: Speaker, into target: Speaker) async {
        let pid = projectID
        await perform("Could not merge the speakers") {
            try await $0.mergeSpeakers(projectId: pid, speakerIds: [source.id], into: target.id)
        }
    }

    func addSpeaker(named name: String) async {
        let pid = projectID
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        await perform("Could not add the speaker") {
            try await $0.createSpeaker(projectId: pid, name: trimmed.isEmpty ? nil : trimmed)
        }
    }

    func assign(_ segmentIDs: [Int], to speaker: Speaker) async {
        let pid = projectID
        await perform("Could not change the speaker") {
            try await $0.assignSpeaker(projectId: pid, segmentIds: segmentIDs, speakerId: speaker.id)
        }
    }

    func assign(_ segmentIDs: [Int], toNewSpeakerNamed name: String) async {
        let pid = projectID
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        await perform("Could not add the speaker") {
            try await $0.assignNewSpeaker(projectId: pid, segmentIds: segmentIDs, name: trimmed.isEmpty ? "New speaker" : trimmed)
        }
    }

    func split(_ segment: Segment, beforeWord wordIndex: Int) async {
        let pid = projectID
        await perform("Could not split the segment") {
            try await $0.splitSegment(projectId: pid, segmentId: segment.id, wordIndex: wordIndex)
        }
    }

    func mergeWithNext(_ segment: Segment) async {
        guard let next = neighbour(of: segment, offset: 1) else { return }
        let pid = projectID
        await perform("Could not merge the segments") {
            try await $0.mergeSegments(projectId: pid, segmentIds: [segment.id, next.id])
        }
    }

    func editText(_ segment: Segment, to text: String) async {
        let pid = projectID
        await perform("Could not change the text") {
            try await $0.editText(projectId: pid, segmentId: segment.id, text: text)
        }
    }

    func markUncertain(_ segment: Segment, _ uncertain: Bool, note: String? = nil) async {
        let pid = projectID
        await perform("Could not mark the passage") {
            try await $0.markUncertain(projectId: pid, segmentId: segment.id, uncertain: uncertain, note: note)
        }
    }

    func undo() async {
        guard history.canUndo else { return }
        let pid = projectID
        await perform("Could not undo") { try await $0.undo(projectId: pid) }
    }

    func redo() async {
        guard history.canRedo else { return }
        let pid = projectID
        await perform("Could not redo") { try await $0.redo(projectId: pid) }
    }
}
