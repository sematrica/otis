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
    /// Find in this transcript (engine full-text search); empty = off.
    @Published var searchText = ""
    /// Passages matching ``searchText`` (nil while no search is active).
    @Published private(set) var searchMatches: Set<Int>?
    /// The searched words, for highlighting.
    @Published private(set) var searchTerms: [String] = []
    /// A passage to scroll into view (from a search result or Find).
    @Published var scrollRequest: Int?
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
        if searchMatches != nil { await runSearch() }  // the text may have changed
    }

    /// Searches this transcript for ``searchText``.
    func runSearch() async {
        let text = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            searchMatches = nil
            searchTerms = []
            return
        }
        do {
            let response = try await client.searchTranscripts(text, projectId: projectID, limit: 500)
            searchMatches = Set(response.results.map(\.segmentId))
            searchTerms = response.terms
        } catch {
            searchMatches = []  // e.g. only punctuation typed: nothing matches
            searchTerms = []
        }
    }

    /// Shows a passage: selected and scrolled into view.
    func focus(on segmentID: Int) {
        guard segments.contains(where: { $0.id == segmentID }) else { return }
        if let matches = searchMatches, !matches.contains(segmentID) {
            searchText = ""
            searchMatches = nil
            searchTerms = []
        }
        if filter != .all && !visibleSegments.contains(where: { $0.id == segmentID }) { filter = .all }
        selectedSegmentID = segmentID
        scrollRequest = segmentID
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
        let shown: [Segment]
        switch filter {
        case .all:
            shown = segments
        case .speaker:
            shown = focusedSpeakerID.map { id in segments.filter { $0.speakerId == id } } ?? segments
        case .review:
            // most severe first, then in time order
            shown = segments.filter(\.needsReview).sorted {
                (Self.rank($0.reviewPriority), $0.start) < (Self.rank($1.reviewPriority), $1.start)
            }
        }
        guard let searchMatches else { return shown }
        return shown.filter { searchMatches.contains($0.id) }
    }

    /// The engine sets each segment's review priority from its flags
    /// (verify/review.py), so the app and the engine always agree.
    static func needsReview(_ segment: Segment) -> Bool { segment.needsReview }

    private static func rank(_ priority: String?) -> Int {
        switch priority {
        case "high": return 0
        case "medium": return 1
        default: return 2
        }
    }

    var reviewCount: Int { segments.filter(\.needsReview).count }

    /// A re-check job of this project is queued or running.
    var isRechecking: Bool {
        app.jobs.values.contains { $0.projectId == projectID && $0.kind == "recheck" && $0.status.isActive }
    }

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

    // MARK: Review (Stage 3)

    func remove(_ segment: Segment) async {
        let pid = projectID
        await perform("Could not remove the passage") { try await $0.deleteSegment(projectId: pid, segmentId: segment.id) }
    }

    func chooseWording(_ segment: Segment, source: String) async {
        let pid = projectID
        await perform("Could not change the wording") {
            try await $0.chooseWording(projectId: pid, segmentId: segment.id, source: source)
        }
    }

    func setReviewed(_ segments: [Segment], _ reviewed: Bool) async {
        let pid = projectID
        let ids = segments.map(\.id)
        guard !ids.isEmpty else { return }
        await perform("Could not update the review state") {
            try await $0.markReviewed(projectId: pid, segmentIds: ids, reviewed: reviewed)
        }
    }

    /// Re-transcribe passages and let the audio decide their wording. Runs as a
    /// job; the transcript reloads when it completes (one undoable change).
    func recheck(_ segments: [Segment]) async {
        let ids = segments.map(\.id)
        guard !ids.isEmpty else { return }
        do {
            _ = try await client.recheck(projectId: projectID, segmentIds: ids)
            lastChange = "Re-checking \(ids.count) passage\(ids.count == 1 ? "" : "s")…"
        } catch {
            app.report(error, title: "Could not re-check")
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
