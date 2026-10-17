import AVKit
import AppKit
import SwiftUI
import VerbatimCore

/// The transcript workspace: speakers (left), the player (centre) and the
/// editable transcript (right). The transcript follows playback, and
/// clicking a segment or word seeks the player to it.
struct ProjectWorkspaceView: View {
    let projectID: String
    @EnvironmentObject private var model: AppModel
    @StateObject private var player = PlayerController()
    @StateObject private var workspace: WorkspaceModel
    @State private var isRenaming = false
    @State private var newTitle = ""
    @State private var confirmDelete = false
    @State private var transcribingAgain = false
    @State private var reviewingWithAI = false

    init(projectID: String) {
        self.projectID = projectID
        _workspace = StateObject(wrappedValue: WorkspaceModel(projectID: projectID, app: AppEnvironment.model))
    }

    var body: some View {
        if let project = model.project(projectID) {
            content(project)
                .navigationTitle(project.title)
                .toolbar { toolbar(project) }
                .task(id: model.transcriptRevision[projectID, default: 0]) {
                    await workspace.refresh(knownRevision: model.knownTranscriptRevision[projectID])
                }
                .onAppear { player.load(project.sourceURL) }
                .onDisappear { player.stop() }
                .focusedSceneValue(\.transcriptCommands, TranscriptCommands(workspace: workspace, player: player))
                .alert("Rename Project", isPresented: $isRenaming) {
                    TextField("Title", text: $newTitle)
                    Button("Rename") { Task { await model.rename(project, to: newTitle) } }
                    Button("Cancel", role: .cancel) {}
                }
                .confirmationDialog("Delete “\(project.title)”?", isPresented: $confirmDelete) {
                    Button("Delete Project", role: .destructive) { Task { await model.delete(project) } }
                } message: {
                    Text("The transcript, your corrections and the engine's working files are deleted. The source media file is not touched.")
                }
                .sheet(isPresented: $transcribingAgain) {
                    TranscribeAgainSheet(project: project, hasCorrections: workspace.history.editCount > 0)
                }
                .sheet(isPresented: $reviewingWithAI) {
                    AIReviewSheet(projectID: projectID, app: model, workspace: workspace) { time in
                        player.play(from: time)
                    }
                    .environmentObject(model)
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
            // Minimum widths fit the smallest window (1020 pt including the
            // ~145 pt sidebar) without clipping the transcript; below ~300 pt
            // the player's controls drop the timeline.
            HSplitView {
                SpeakersPanel(workspace: workspace, player: player)
                    .frame(minWidth: 170, idealWidth: 230, maxWidth: 340)
                PlayerPanel(project: project, workspace: workspace, player: player)
                    .frame(minWidth: 300, idealWidth: 440)
                TranscriptPanel(project: project, workspace: workspace, player: player)
                    .frame(minWidth: 380, idealWidth: 540)
            }
        }
    }

    @ToolbarContentBuilder
    private func toolbar(_ project: Project) -> some ToolbarContent {
        ToolbarItemGroup {
            Menu {
                ForEach(ExportFormat.allCases) { format in
                    Button(format.displayName) { Task { await model.export(project, format: format) } }
                }
                Button("CSV, One Row per Word") {
                    Task { await model.export(project, format: .csv, options: ExportOptions(words: true)) }
                }
            } label: {
                Label("Export", systemImage: "square.and.arrow.up")
            }
            .disabled(project.segmentCount == 0)
            .help("Export the transcript as text, subtitles (SRT, WebVTT), JSON, CSV or Markdown")

            Button {
                transcribingAgain = true
            } label: {
                Label("Transcribe Again", systemImage: "arrow.clockwise")
            }
            .disabled(model.latestJob(for: project.id)?.status.isActive == true)
            .help("Run a new transcription (replaces the current transcript and corrections when it completes)")

            Menu {
                Button("Show Passages Needing Review (\(workspace.reviewCount))") { workspace.filter = .review }
                    .disabled(workspace.reviewCount == 0)
                Button("Re-check Passages Needing Review") {
                    Task { await workspace.recheck(workspace.segments.filter(\.needsReview)) }
                }
                .disabled(workspace.reviewCount == 0 || model.latestJob(for: project.id)?.status.isActive == true)
                .help("Decode these passages again and keep the wording the audio supports best")
                Button("Mark Shown Passages as Reviewed") {
                    Task { await workspace.setReviewed(workspace.visibleSegments.filter { !$0.isReviewed }, true) }
                }
                .disabled(workspace.visibleSegments.allSatisfy(\.isReviewed))
                Divider()
                Button("Review with AI…") { reviewingWithAI = true }
                    .disabled(project.segmentCount == 0)
            } label: {
                Label("Review", systemImage: "checklist")
            }
            .help("Review doubtful passages")

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
}

/// Centre panel: playback and what is being said now.
private struct PlayerPanel: View {
    let project: Project
    @ObservedObject var workspace: WorkspaceModel
    @ObservedObject var player: PlayerController

    var body: some View {
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
            nowSpeaking
                .padding(.horizontal, 12)
                .padding(.top, 10)
            MediaFacts(project: project)
                .padding(12)
        }
    }

    @ViewBuilder
    private var nowSpeaking: some View {
        if let id = workspace.activeSegmentID(at: player.currentTime), let segment = workspace.segment(id) {
            let speaker = workspace.speaker(segment.speakerId)
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                SpeakerDot(speaker: speaker, size: 10)
                VStack(alignment: .leading, spacing: 2) {
                    Text(speaker?.displayName ?? "Unknown speaker").font(.callout.weight(.semibold))
                    Text(segment.text.trimmingCharacters(in: .whitespaces))
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .lineLimit(3)
                }
                Spacer(minLength: 0)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

/// Re-run transcription with chosen options; warns that corrections will be replaced.
private struct TranscribeAgainSheet: View {
    let project: Project
    let hasCorrections: Bool
    @EnvironmentObject private var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var mode = "balanced"
    @State private var speakers = SpeakerChoice()

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Transcribe “\(project.title)” Again").font(.headline)
            Picker("Mode", selection: $mode) {
                ForEach(model.modes) { mode in Text(mode.label).tag(mode.id) }
            }
            .pickerStyle(.segmented)
            if let selected = model.modes.first(where: { $0.id == mode }) {
                Text(selected.description).font(.caption).foregroundStyle(.secondary)
            }
            SpeakerOptionsView(choice: $speakers)
            if hasCorrections {
                Label("Your corrections to the current transcript are replaced when the new transcription completes.",
                      systemImage: "exclamationmark.triangle")
                    .font(.callout)
                    .foregroundStyle(.orange)
            }
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button("Transcribe") {
                    let options = JobOptions(mode: mode, diarize: speakers.tristate, speakers: speakers.bounds)
                    Task { await model.retranscribe(project, options: options) }
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(18)
        .frame(width: 520)
    }
}

struct JobBanner: View {
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
        case .running: return job.runningStage?.label ?? "Transcribing"
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

// MARK: - Menu commands

/// What the Edit and Playback menus act on while a workspace is frontmost.
struct TranscriptCommands {
    let workspace: WorkspaceModel
    let player: PlayerController
}

struct TranscriptCommandsKey: FocusedValueKey {
    typealias Value = TranscriptCommands
}

extension FocusedValues {
    var transcriptCommands: TranscriptCommands? {
        get { self[TranscriptCommandsKey.self] }
        set { self[TranscriptCommandsKey.self] = newValue }
    }
}

/// Undo/Redo for transcript corrections. While text is being typed (a text
/// field or editor has focus) the commands go to that text instead.
@MainActor
struct TranscriptMenuCommands: Commands {
    @FocusedValue(\.transcriptCommands) private var commands

    var body: some Commands {
        CommandGroup(replacing: .undoRedo) {
            Button("Undo") { perform(undo: true) }
                .keyboardShortcut("z", modifiers: .command)
            Button("Redo") { perform(undo: false) }
                .keyboardShortcut("z", modifiers: [.command, .shift])
        }
        CommandMenu("Playback") {
            Button("Play / Pause") { commands?.player.togglePlayback() }
                .keyboardShortcut(.space, modifiers: .option)
                .disabled(commands == nil)
        }
    }

    private func perform(undo: Bool) {
        if let text = NSApp.keyWindow?.firstResponder as? NSText {
            if undo { text.undoManager?.undo() } else { text.undoManager?.redo() }
            return
        }
        guard let workspace = commands?.workspace else { return }
        Task {
            if undo { await workspace.undo() } else { await workspace.redo() }
        }
    }
}
