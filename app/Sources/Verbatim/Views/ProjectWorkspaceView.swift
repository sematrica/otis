import AVKit
import AppKit
import SwiftUI
import VerbatimCore

/// Stage 1 workspace: player and timestamped transcript. Clicking a segment
/// seeks the player to it. (Speaker panel and editing arrive in Stage 2.)
struct ProjectWorkspaceView: View {
    let projectID: String
    @EnvironmentObject private var model: AppModel
    @StateObject private var player = PlayerController()
    @State private var transcript: Transcript?
    @State private var loadError: String?
    @State private var isRenaming = false
    @State private var newTitle = ""
    @State private var confirmDelete = false

    var body: some View {
        if let project = model.project(projectID) {
            content(project)
                .navigationTitle(project.title)
                .toolbar { toolbar(project) }
                .task(id: model.transcriptRevision[projectID, default: 0]) { await loadTranscript(project) }
                .onAppear { player.load(project.sourceURL) }
                .onDisappear { player.stop() }
                .alert("Rename Project", isPresented: $isRenaming) {
                    TextField("Title", text: $newTitle)
                    Button("Rename") { Task { await model.rename(project, to: newTitle) } }
                    Button("Cancel", role: .cancel) {}
                }
                .confirmationDialog("Delete “\(project.title)”?", isPresented: $confirmDelete) {
                    Button("Delete Project", role: .destructive) { Task { await model.delete(project) } }
                } message: {
                    Text("The transcript and the engine's working files are deleted. The source media file is not touched.")
                }
        } else {
            EmptyState(title: "Project not found", systemImage: "questionmark.folder", message: nil)
        }
    }

    @ViewBuilder
    private func content(_ project: Project) -> some View {
        VStack(spacing: 0) {
            if let job = model.latestJob(for: project.id), job.status != .completed {
                JobBanner(job: job)
                Divider()
            }
            HSplitView {
                playerPanel(project)
                    .frame(minWidth: 320, idealWidth: 480)
                TranscriptPanel(transcript: transcript, loadError: loadError, project: project) { seconds in
                    player.play(from: seconds)
                }
                .frame(minWidth: 380, idealWidth: 520)
            }
        }
    }

    @ViewBuilder
    private func playerPanel(_ project: Project) -> some View {
        VStack(spacing: 0) {
            if let avPlayer = player.player {
                PlayerView(player: avPlayer)
                    .frame(minHeight: project.hasVideo ? 240 : 80)
            } else {
                EmptyState(title: "Playback unavailable", systemImage: "play.slash", message: player.loadError)
            }
            if !project.hasVideo, player.player != nil {
                Text("Audio only").font(.caption).foregroundStyle(.secondary).padding(6)
            }
            MediaFacts(project: project)
                .padding(12)
        }
    }

    @ToolbarContentBuilder
    private func toolbar(_ project: Project) -> some ToolbarContent {
        ToolbarItemGroup {
            Button {
                Task { await model.exportText(project) }
            } label: {
                Label("Export TXT", systemImage: "square.and.arrow.up")
            }
            .disabled(project.segmentCount == 0)
            .help("Export the transcript as plain text")

            Menu {
                ForEach(model.modes) { mode in
                    Button(mode.label) {
                        Task { await model.retranscribe(project, options: JobOptions(mode: mode.id)) }
                    }
                }
            } label: {
                Label("Transcribe Again", systemImage: "arrow.clockwise")
            }
            .disabled(model.latestJob(for: project.id)?.status.isActive == true)
            .help("Run a new transcription (replaces the current transcript when it completes)")

            Menu {
                Button("Rename…") { newTitle = project.title; isRenaming = true }
                Button("Show Source in Finder") {
                    if let url = project.sourceURL { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                }
                Divider()
                Button("Delete Project…", role: .destructive) { confirmDelete = true }
            } label: {
                Label("More", systemImage: "ellipsis.circle")
            }
        }
    }

    private func loadTranscript(_ project: Project) async {
        do {
            transcript = try await model.transcript(for: project)
            loadError = nil
        } catch {
            loadError = error.localizedDescription
        }
    }
}

private struct JobBanner: View {
    let job: Job
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title).font(.headline)
                Spacer()
                JobElapsed(job: job)
                if job.status.isActive {
                    Button("Cancel") { Task { await model.cancel(job) } }.disabled(job.cancelRequested)
                }
                if job.status.isResumable {
                    Button("Resume") { Task { await model.resume(job) } }
                }
            }
            .controlSize(.small)
            JobStagesView(job: job)
        }
        .padding(14)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var title: String {
        switch job.status {
        case .queued: return "Queued"
        case .running: return "Transcribing"
        case .failed: return "Transcription failed"
        case .cancelled: return "Transcription cancelled"
        case .interrupted: return "Transcription interrupted"
        case .completed: return "Completed"
        }
    }
}

private struct MediaFacts: View {
    let project: Project

    var body: some View {
        Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
            GridRow {
                Text("Duration").foregroundStyle(.secondary)
                Text(Timecode.duration(project.duration))
            }
            GridRow {
                Text("Format").foregroundStyle(.secondary)
                Text(project.media.formatLongName).lineLimit(1)
            }
            if let audio = project.media.audioStreams.first(where: { $0.index == project.media.selectedAudioStream }) {
                GridRow {
                    Text("Audio").foregroundStyle(.secondary)
                    Text("\(audio.codec), \(audio.sampleRate) Hz, \(audio.channels) ch")
                }
            }
            if let video = project.media.videoStreams.first {
                GridRow {
                    Text("Video").foregroundStyle(.secondary)
                    Text("\(video.codec), \(video.width)×\(video.height)")
                }
            }
            GridRow {
                Text("File").foregroundStyle(.secondary)
                Text(project.source.path ?? "—").lineLimit(1).truncationMode(.middle).textSelection(.enabled)
            }
        }
        .font(.caption)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct TranscriptPanel: View {
    let transcript: Transcript?
    let loadError: String?
    let project: Project
    let onSeek: (Double) -> Void

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if let loadError {
                EmptyState(title: "Could not load the transcript", systemImage: "exclamationmark.triangle", message: loadError)
            } else if let transcript, !transcript.segments.isEmpty {
                List(transcript.segments) { segment in
                    SegmentRow(segment: segment) { onSeek(segment.start) }
                }
                .listStyle(.inset)
            } else if transcript != nil {
                EmptyState(title: "No transcript yet", systemImage: "text.bubble",
                           message: project.status == .ready
                               ? "No speech was detected in this recording."
                               : "The transcript appears here when processing completes.")
            } else {
                ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }

    private var header: some View {
        HStack(spacing: 10) {
            Text("Transcript").font(.headline)
            if let language = transcript?.language {
                Text(language.uppercased()).font(.caption.weight(.semibold))
                    .padding(.horizontal, 6).padding(.vertical, 1)
                    .background(Color.secondary.opacity(0.15), in: Capsule())
            }
            Spacer()
            if let effective = transcript?.effective {
                EffectiveSummary(effective: effective)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }
}

/// Which model and backend actually produced the transcript.
private struct EffectiveSummary: View {
    let effective: JSONValue
    @EnvironmentObject private var model: AppModel
    @State private var showDetails = false

    var body: some View {
        Button {
            showDetails.toggle()
        } label: {
            Text(summary).font(.caption).foregroundStyle(.secondary)
        }
        .buttonStyle(.plain)
        .help("Settings that produced this transcript")
        .popover(isPresented: $showDetails) {
            ScrollView {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(rows) { row in
                        HStack(alignment: .top) {
                            Text(row.key.humanizedKey).foregroundStyle(.secondary).frame(width: 170, alignment: .leading)
                            Text(row.value).textSelection(.enabled)
                        }
                        .font(.caption)
                    }
                }
                .padding(14)
            }
            .frame(width: 460, height: 340)
        }
    }

    private var summary: String {
        let modelName = effective.lookup("model")?.lookup("display_name")?.stringValue ?? "Unknown model"
        let backendID = effective.lookup("backend")?.stringValue ?? ""
        let backend = model.engineInfo?.backends.first { $0.name == backendID }?.displayName ?? backendID
        return backend.isEmpty ? modelName : "\(modelName) · \(backend)"
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

private struct SegmentRow: View {
    let segment: Segment
    let onSeek: () -> Void

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Button(action: onSeek) {
                Text(Timecode.clock(segment.start))
                    .font(.callout.monospacedDigit())
                    .foregroundStyle(Color.accentColor)
            }
            .buttonStyle(.plain)
            .help("Play from here")
            Text(segment.text)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            if let note = boundaryNote {
                Image(systemName: "scissors")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .help(note)
            }
        }
        .padding(.vertical, 3)
        .contentShape(Rectangle())
        .onTapGesture(count: 2, perform: onSeek)
    }

    /// Automatic decisions made while joining chunks, shown for transparency.
    private var boundaryNote: String? {
        let types = Set(segment.flagTypes)
        if types.contains("boundary_duplicate_removed") {
            let removed = segment.flags.first { $0["type"]?.stringValue == "boundary_duplicate_removed" }
            let text = removed?["text"]?.stringValue ?? ""
            return "A repeated passage at a chunk boundary was removed: “\(text)”."
        }
        if types.contains("trimmed_at_chunk_boundary") {
            return "This segment was joined at a processing-chunk boundary."
        }
        return nil
    }
}
