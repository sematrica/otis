import AppKit
import SwiftUI
import UniformTypeIdentifiers
import VerbatimCore

struct LibraryView: View {
    let onOpen: (String) -> Void
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            ImportPanel()
                .padding(20)
            Divider()
            if model.projects.isEmpty {
                EmptyState(title: "Your library is empty", systemImage: "books.vertical",
                           message: "Imported recordings and their transcripts appear here.")
            } else {
                List(model.projects) { project in
                    Button { onOpen(project.id) } label: { ProjectRow(project: project) }
                        .buttonStyle(.plain)
                        .contextMenu {
                            Button("Open") { onOpen(project.id) }
                            Button("Show Source in Finder") {
                                if let url = project.sourceURL { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                            }
                            Divider()
                            Button("Delete Project", role: .destructive) { Task { await model.delete(project) } }
                        }
                }
            }
        }
        .navigationTitle("Library")
    }
}

/// Drag-and-drop import, processing-mode selection and the Transcribe button.
struct ImportPanel: View {
    @EnvironmentObject private var model: AppModel
    @State private var staged: [StagedFile] = []
    @State private var isTargeted = false
    @State private var isStaging = false
    @State private var mode = "balanced"
    @State private var backend = "auto"
    @State private var language = ""
    @State private var speakers = SpeakerChoice()
    @State private var preflight: Result<Preflight, EngineClientError>?
    @State private var pendingDownload: ModelEntry?
    @State private var isStarting = false
    @State private var youtubeLink = ""
    @State private var checkingLink: String?

    private var options: JobOptions {
        JobOptions(mode: mode, backend: backend, language: language.isEmpty ? nil : language,
                   diarize: speakers.tristate, speakers: speakers.bounds)
    }

    private var validFiles: [StagedFile] { staged.filter { $0.error == nil && $0.media != nil } }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            dropZone
            if !staged.isEmpty { stagedList }
            youtubeRow
            HStack(alignment: .top, spacing: 24) {
                modePicker
                Spacer(minLength: 0)
                advanced
            }
            SpeakerOptionsView(choice: $speakers)
            HStack {
                preflightStatus
                Spacer()
                Button {
                    Task { await startTranscription() }
                } label: {
                    Label("Transcribe", systemImage: "text.bubble")
                        .frame(minWidth: 110)
                }
                .keyboardShortcut(.defaultAction)
                .controlSize(.large)
                .disabled(validFiles.isEmpty || !preflightOK || isStarting)
            }
        }
        .sheet(isPresented: Binding(get: { checkingLink != nil }, set: { if !$0 { checkingLink = nil } })) {
            if let link = checkingLink {
                YouTubeImportSheet(link: link, options: options) { _ in youtubeLink = "" }
                    .environmentObject(model)
            }
        }
        .task(id: options) { await runPreflight() }
        .onChange(of: model.models) { _ in Task { await runPreflight() } }
        .confirmationDialog(
            pendingDownload.map { downloadTitle($0) } ?? "",
            isPresented: Binding(get: { pendingDownload != nil }, set: { if !$0 { pendingDownload = nil } }),
            titleVisibility: .visible,
            presenting: pendingDownload
        ) { entry in
            Button("Download \(ByteSize.string(entry.approxDownloadBytes))") { Task { await model.download(entry) } }
            Button("Cancel", role: .cancel) {}
        } message: { entry in
            Text(downloadMessage(entry, modelsDir: model.models?.modelsDir))
        }
    }

    // MARK: Drop zone

    private var dropZone: some View {
        RoundedRectangle(cornerRadius: 12)
            .strokeBorder(style: StrokeStyle(lineWidth: 1.5, dash: [6, 4]))
            .foregroundStyle(isTargeted ? Color.accentColor : Color.secondary.opacity(0.5))
            .background(RoundedRectangle(cornerRadius: 12).fill(isTargeted ? Color.accentColor.opacity(0.08) : Color.clear))
            .frame(height: 110)
            .overlay {
                VStack(spacing: 8) {
                    if isStaging {
                        ProgressView().controlSize(.small)
                        Text("Checking files…").foregroundStyle(.secondary)
                    } else {
                        Image(systemName: "square.and.arrow.down.on.square").font(.title2).foregroundStyle(.secondary)
                        Text("Drop audio or video files here").font(.headline)
                        Button("Choose Files…", action: chooseFiles)
                    }
                }
            }
            .dropDestination(for: URL.self) { urls, _ in
                let files = urls.filter(\.isFileURL)
                guard !files.isEmpty else { return false }
                Task { await stage(files) }
                return true
            } isTargeted: { isTargeted = $0 }
    }

    private var stagedList: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(staged) { file in
                HStack(spacing: 8) {
                    Image(systemName: file.error == nil ? (file.media?.hasVideo == true ? "film" : "waveform") : "xmark.octagon")
                        .foregroundStyle(file.error == nil ? Color.secondary : Color.red)
                    Text(file.url.lastPathComponent).lineLimit(1)
                    if let media = file.media {
                        Text("\(Timecode.duration(media.duration)) · \(media.formatName)")
                            .foregroundStyle(.secondary)
                    }
                    if let error = file.error {
                        Text(error).foregroundStyle(.red).lineLimit(2)
                    }
                    Spacer()
                    Button {
                        staged.removeAll { $0.id == file.id }
                    } label: { Image(systemName: "xmark.circle.fill") }
                        .buttonStyle(.borderless)
                        .foregroundStyle(.secondary)
                }
                .font(.callout)
            }
        }
    }

    /// A YouTube link, imported with the settings below once checked.
    private var youtubeRow: some View {
        HStack(spacing: 8) {
            Image(systemName: "play.rectangle").foregroundStyle(.secondary)
            TextField("Or paste a YouTube link (a video you are authorized to download)", text: $youtubeLink)
                .textFieldStyle(.roundedBorder)
                .onSubmit { checkLink() }
            Button("Check Link…", action: checkLink)
                .disabled(youtubeLink.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !preflightOK)
                .help(preflightOK ? "Shows the video, its licence and captions before anything is retrieved"
                                  : "Choose settings that can run first (see below)")
        }
    }

    private func checkLink() {
        let link = youtubeLink.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !link.isEmpty, preflightOK else { return }
        checkingLink = link
    }

    // MARK: Options

    private var modePicker: some View {
        VStack(alignment: .leading, spacing: 6) {
            Picker("Mode", selection: $mode) {
                ForEach(model.modes) { mode in Text(mode.label).tag(mode.id) }
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 420)
            if let selected = model.modes.first(where: { $0.id == mode }) {
                Text(selected.description).font(.caption).foregroundStyle(.secondary)
                if !selected.limitations.isEmpty {
                    Label(selected.limitations, systemImage: "info.circle")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
            }
        }
    }

    private var advanced: some View {
        VStack(alignment: .trailing, spacing: 6) {
            Picker("Engine", selection: $backend) {
                Text("Automatic").tag("auto")
                ForEach(model.engineInfo?.backends ?? []) { backend in
                    Text(backend.displayName).tag(backend.name).disabled(!backend.available)
                }
            }
            .frame(width: 240)
            Picker("Language", selection: $language) {
                Text("Detect automatically").tag("")
                ForEach(Languages.common, id: \.code) { lang in Text(lang.name).tag(lang.code) }
            }
            .frame(width: 240)
        }
    }

    @ViewBuilder
    private var preflightStatus: some View {
        switch preflight {
        case .none:
            EmptyView()
        case .some(.success(let result)):
            VStack(alignment: .leading, spacing: 2) {
                Label("\(result.resolved.model.displayName) · \(backendName(result.resolved.backend))", systemImage: "cpu")
                    .font(.callout)
                ForEach(result.resolved.warnings.filter { !$0.contains("not implemented yet") }, id: \.self) { warning in
                    Text(warning).font(.caption).foregroundStyle(.secondary)
                }
                if let diarization = result.resolved.diarization, diarization.enabled {
                    Label("Speakers identified on this Mac", systemImage: "person.2").font(.caption).foregroundStyle(.secondary)
                }
                alignmentStatus(result.resolved.alignment)
                if let recheck = result.resolved.recheck, recheck.enabled {
                    Label(recheck.acousticModels.isEmpty
                              ? "Doubtful passages re-transcribed; with no word-alignment model, other wordings are listed for review"
                              : "Doubtful passages re-transcribed and compared against the audio",
                          systemImage: "checkmark.seal")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
        case .some(.failure(let error)):
            HStack(spacing: 8) {
                Label(error.localizedDescription, systemImage: "exclamationmark.triangle")
                    .font(.callout)
                    .foregroundStyle(.orange)
                if error.engineCode == "model_not_installed", let entry = suggestedModel(for: error) {
                    if let state = model.downloads[entry.id], !state.isFinished {
                        DownloadProgress(state: state)
                    } else {
                        Button("Download…") { pendingDownload = entry }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func alignmentStatus(_ alignment: Preflight.Alignment?) -> some View {
        if let alignment {
            if alignment.enabled {
                Label("Word times aligned to the audio", systemImage: "waveform.badge.magnifyingglass")
                    .font(.caption).foregroundStyle(.secondary)
            } else if model.modes.first(where: { $0.id == mode })?.align ?? false {
                HStack(spacing: 6) {
                    Label("Word times from Whisper: no word-alignment model is installed", systemImage: "info.circle")
                        .font(.caption).foregroundStyle(.secondary)
                    if let entry = suggestedAligner {
                        if let state = model.downloads[entry.id], !state.isFinished {
                            DownloadProgress(state: state)
                        } else {
                            Button("Download…") { pendingDownload = entry }.controlSize(.small)
                        }
                    }
                }
            }
        }
    }

    /// The alignment model that fits the chosen language.
    private var suggestedAligner: ModelEntry? {
        let id = (language.isEmpty || language == "en") ? "alignment/en-conformer-small" : "alignment/mms-300m"
        return model.modelEntry(id)
    }

    private var preflightOK: Bool {
        if case .some(.success) = preflight { return true }
        return false
    }

    private func backendName(_ name: String) -> String {
        model.engineInfo?.backends.first { $0.name == name }?.displayName ?? name
    }

    private func suggestedModel(for error: EngineClientError) -> ModelEntry? {
        guard case .engine(_, _, let data) = error, let id = data?.lookup("model_id")?.stringValue else { return nil }
        return model.models?.models.first { $0.id == id }
    }

    // MARK: Actions

    private func chooseFiles() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        // Any file may be chosen: the engine validates the actual contents
        // (some containers such as MKV have no system content type).
        panel.allowedContentTypes = [.audiovisualContent, .audio, .movie, .item]
        guard panel.runModal() == .OK else { return }
        Task { await stage(panel.urls) }
    }

    private func stage(_ urls: [URL]) async {
        isStaging = true
        let newFiles = await model.stage(urls)
        staged.append(contentsOf: newFiles)
        isStaging = false
    }

    private func runPreflight() async {
        preflight = await model.preflight(options)
    }

    private func startTranscription() async {
        isStarting = true
        defer { isStarting = false }
        if await model.transcribe(validFiles, options: options) {
            staged.removeAll()
            model.section = .processing
        }
    }
}

/// Confirmation texts for an explicit model download.
func downloadTitle(_ entry: ModelEntry) -> String {
    switch entry.modelKind {
    case "asr": return "Download \(entry.displayName) for \(entry.backend == "mlx" ? "MLX" : "faster-whisper")?"
    default: return "Download \(entry.displayName)?"
    }
}

func downloadMessage(_ entry: ModelEntry, modelsDir: String?) -> String {
    var text = "About \(ByteSize.string(entry.approxDownloadBytes)) will be downloaded from \(entry.sourceDescription) "
        + "into \(modelsDir ?? "the models folder")."
    if let license = entry.license, !license.isEmpty { text += " Licence: \(license)." }
    if let files = entry.files, !files.isEmpty { text += " The files are checked against published checksums." }
    return text + " Nothing is downloaded without your confirmation."
}

struct DownloadProgress: View {
    let state: ModelDownload

    var body: some View {
        HStack(spacing: 6) {
            if let done = state.done, let total = state.total, total > 0, state.totalIsEstimate != true {
                ProgressView(value: Double(done), total: Double(total)).frame(width: 120)
                Text("\(ByteSize.string(done)) of \(ByteSize.string(total))").font(.caption).monospacedDigit()
            } else {
                ProgressView().controlSize(.small)
                Text(state.status == "cancelling" ? "Cancelling…" : "Starting download…").font(.caption)
            }
        }
    }
}

enum Languages {
    struct Language { let code: String; let name: String }
    static let common: [Language] = [
        ("en", "English"), ("es", "Spanish"), ("fr", "French"), ("de", "German"), ("it", "Italian"),
        ("pt", "Portuguese"), ("nl", "Dutch"), ("ru", "Russian"), ("uk", "Ukrainian"), ("pl", "Polish"),
        ("tr", "Turkish"), ("ar", "Arabic"), ("he", "Hebrew"), ("hi", "Hindi"), ("ja", "Japanese"),
        ("ko", "Korean"), ("zh", "Chinese"), ("uz", "Uzbek"), ("kk", "Kazakh"), ("sv", "Swedish"),
    ].map { Language(code: $0.0, name: $0.1) }
}
