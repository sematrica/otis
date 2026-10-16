import AppKit
import SwiftUI
import VerbatimCore

/// Right panel: the timestamped, speaker-labelled, editable transcript. The
/// segment being played is highlighted (and scrolled into view while
/// "Follow playback" is on); clicking a segment or a word seeks to it.
struct TranscriptPanel: View {
    let project: Project
    @ObservedObject var workspace: WorkspaceModel
    @ObservedObject var player: PlayerController
    @State private var follow = true
    @State private var editing: Segment?
    @State private var newSpeakerFor: [Int]?
    @State private var markingSegment: Segment?
    @State private var showingOriginal: Segment?

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            content
        }
        .sheet(item: $editing) { segment in
            EditTextSheet(segment: segment) { text in Task { await workspace.editText(segment, to: text) } }
        }
        .sheet(isPresented: Binding(get: { newSpeakerFor != nil }, set: { if !$0 { newSpeakerFor = nil } })) {
            NameSheet(title: "New Speaker", prompt: "Assign the segment to a new speaker named:", initial: "",
                      action: "Assign") { name in
                if let ids = newSpeakerFor { Task { await workspace.assign(ids, toNewSpeakerNamed: name) } }
            }
        }
        .sheet(item: $markingSegment) { segment in
            NameSheet(title: "Mark as Uncertain", prompt: "Optional note (what needs checking):", initial: "",
                      action: "Mark") { note in
                Task { await workspace.markUncertain(segment, true, note: note.isEmpty ? nil : note) }
            }
        }
        .alert("Original text", isPresented: Binding(get: { showingOriginal != nil }, set: { if !$0 { showingOriginal = nil } }),
               presenting: showingOriginal) { _ in
            Button("OK", role: .cancel) {}
        } message: { segment in
            Text("As recognised: “\(segment.originalText.trimmingCharacters(in: .whitespaces))”")
        }
    }

    @ViewBuilder
    private var content: some View {
        if let error = workspace.loadError {
            EmptyState(title: "Could not load the transcript", systemImage: "exclamationmark.triangle", message: error)
        } else if let transcript = workspace.transcript, !transcript.segments.isEmpty {
            let active = workspace.activeSegmentID(at: player.currentTime)
            ScrollViewReader { proxy in
                List(workspace.visibleSegments) { segment in
                    SegmentRow(
                        segment: segment,
                        speaker: workspace.speaker(segment.speakerId),
                        originalSpeaker: segment.isSpeakerEdited ? workspace.speaker(segment.originalSpeakerId) : nil,
                        speakers: workspace.speakers,
                        isActive: segment.id == active,
                        isSelected: segment.id == workspace.selectedSegmentID,
                        playbackTime: segment.id == active ? player.currentTime : nil,
                        actions: actions(for: segment)
                    )
                    .id(segment.id)
                    .listRowBackground(rowBackground(segment, active: active))
                }
                .listStyle(.inset)
                .onChange(of: active) { id in
                    guard follow, player.isPlaying, let id else { return }
                    withAnimation(.easeInOut(duration: 0.25)) { proxy.scrollTo(id, anchor: .center) }
                }
            }
        } else if workspace.transcript != nil {
            EmptyState(title: "No transcript yet", systemImage: "text.bubble",
                       message: project.status == .ready
                           ? "No speech was detected in this recording."
                           : "The transcript appears here when processing completes.")
        } else {
            ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private func rowBackground(_ segment: Segment, active: Int?) -> some View {
        Group {
            if segment.id == active {
                Color.accentColor.opacity(0.16)
            } else if segment.id == workspace.selectedSegmentID {
                Color.secondary.opacity(0.10)
            } else {
                Color.clear
            }
        }
    }

    private var header: some View {
        HStack(spacing: 8) {
            // Title and language never wrap; the model summary below them
            // truncates first when the panel is narrow.
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 6) {
                    Text("Transcript").font(.headline)
                    if let language = workspace.transcript?.language {
                        Text(language.uppercased()).font(.caption.weight(.semibold))
                            .padding(.horizontal, 6).padding(.vertical, 1)
                            .background(Color.secondary.opacity(0.15), in: Capsule())
                    }
                }
                .fixedSize()
                if let effective = workspace.transcript?.effective {
                    EffectiveSummary(effective: effective)
                }
            }
            Spacer(minLength: 4)
            Picker("Show", selection: $workspace.filter) {
                Text("All").tag(WorkspaceModel.Filter.all)
                Text(workspace.speaker(workspace.focusedSpeakerID)?.displayName ?? "Selected speaker")
                    .tag(WorkspaceModel.Filter.speaker)
                Text("Needs review (\(workspace.reviewCount))").tag(WorkspaceModel.Filter.review)
            }
            .labelsHidden()
            .frame(width: 130)
            .help("Show all segments, one speaker's, or the ones flagged for review")
            Toggle(isOn: $follow) { Image(systemName: "arrow.down.to.line") }
                .toggleStyle(.button)
                .help("Follow playback")
            HistoryButtons(workspace: workspace)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    private func actions(for segment: Segment) -> SegmentActions {
        SegmentActions(
            seek: { time in
                workspace.selectedSegmentID = segment.id
                player.play(from: time)
            },
            playSegment: {
                workspace.selectedSegmentID = segment.id
                player.play(from: segment.start, to: segment.end)
            },
            select: { workspace.selectedSegmentID = segment.id },
            assign: { speaker in Task { await workspace.assign([segment.id], to: speaker) } },
            assignNew: { newSpeakerFor = [segment.id] },
            split: { index in Task { await workspace.split(segment, beforeWord: index) } },
            mergeWithNext: workspace.neighbour(of: segment, offset: 1) == nil
                ? nil : { Task { await workspace.mergeWithNext(segment) } },
            edit: { editing = segment },
            markUncertain: { markingSegment = segment },
            clearUncertain: { Task { await workspace.markUncertain(segment, false) } },
            showOriginal: { showingOriginal = segment }
        )
    }
}

/// Undo / redo of corrections, with the action's name.
struct HistoryButtons: View {
    @ObservedObject var workspace: WorkspaceModel

    var body: some View {
        HStack(spacing: 2) {
            Button {
                Task { await workspace.undo() }
            } label: { Image(systemName: "arrow.uturn.backward") }
                .disabled(!workspace.history.canUndo || workspace.isWorking)
                .help(workspace.history.undoSummary.map { "Undo: \($0)" } ?? "Nothing to undo")
            Button {
                Task { await workspace.redo() }
            } label: { Image(systemName: "arrow.uturn.forward") }
                .disabled(!workspace.history.canRedo || workspace.isWorking)
                .help(workspace.history.redoSummary.map { "Redo: \($0)" } ?? "Nothing to redo")
        }
        .buttonStyle(.borderless)
    }
}

/// Which model and backend actually produced the transcript.
struct EffectiveSummary: View {
    let effective: JSONValue
    @EnvironmentObject private var model: AppModel
    @State private var showDetails = false

    var body: some View {
        Button {
            showDetails.toggle()
        } label: {
            Text(summary).font(.caption).foregroundStyle(.secondary).lineLimit(1)
        }
        .buttonStyle(.plain)
        .help("Settings that produced this transcript")
        .popover(isPresented: $showDetails) {
            ScrollView {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(rows) { row in
                        HStack(alignment: .top) {
                            Text(row.key.humanizedKey).foregroundStyle(.secondary).frame(width: 190, alignment: .leading)
                            Text(row.value).textSelection(.enabled)
                        }
                        .font(.caption)
                    }
                }
                .padding(14)
            }
            .frame(width: 520, height: 380)
        }
    }

    private var summary: String {
        let modelName = effective.lookup("model")?.lookup("display_name")?.stringValue ?? "Unknown model"
        let backendID = effective.lookup("backend")?.stringValue ?? ""
        let backend = model.engineInfo?.backends.first { $0.name == backendID }?.displayName ?? backendID
        var text = backend.isEmpty ? modelName : "\(modelName) · \(backend)"
        if effective.lookup("alignment")?.lookup("enabled") == .bool(true) {
            text += " · aligned words"
        }
        return text
    }

    private struct Row: Identifiable {
        let key: String
        let value: String
        var id: String { key }
    }

    private var rows: [Row] {
        var out: [Row] = []
        func walk(_ value: JSONValue, prefix: String) {
            if case .object(let object) = value {
                for key in object.keys.sorted() {
                    walk(object[key]!, prefix: prefix.isEmpty ? key : "\(prefix) › \(key)")
                }
            } else {
                out.append(Row(key: prefix, value: value.displayString))
            }
        }
        walk(effective, prefix: "")
        return out
    }
}

// MARK: - Sheets

struct EditTextSheet: View {
    let segment: Segment
    let onSave: (String) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var text: String

    init(segment: Segment, onSave: @escaping (String) -> Void) {
        self.segment = segment
        self.onSave = onSave
        _text = State(initialValue: segment.text.trimmingCharacters(in: .whitespaces))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Edit Text at \(Timecode.clock(segment.start))").font(.headline)
            TextEditor(text: $text)
                .font(.body)
                .frame(minWidth: 480, minHeight: 120)
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(Color.secondary.opacity(0.3)))
            VStack(alignment: .leading, spacing: 2) {
                Text("As recognised").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Text(segment.originalText.trimmingCharacters(in: .whitespaces))
                    .font(.callout).foregroundStyle(.secondary).textSelection(.enabled)
            }
            Text("The recognised text is kept unchanged alongside your correction. Word times are kept for words you "
                 + "did not change; new words get estimated times.")
                .font(.caption).foregroundStyle(.secondary)
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button("Save") {
                    onSave(text)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(18)
    }
}

/// A one-line text prompt (speaker names, notes).
struct NameSheet: View {
    let title: String
    let prompt: String
    let action: String
    let onDone: (String) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var value: String

    init(title: String, prompt: String, initial: String, action: String, onDone: @escaping (String) -> Void) {
        self.title = title
        self.prompt = prompt
        self.action = action
        self.onDone = onDone
        _value = State(initialValue: initial)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title).font(.headline)
            Text(prompt).font(.callout).foregroundStyle(.secondary)
            TextField("", text: $value)
                .textFieldStyle(.roundedBorder)
                .frame(minWidth: 320)
                .onSubmit { done() }
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button(action) { done() }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(18)
    }

    private func done() {
        onDone(value)
        dismiss()
    }
}
