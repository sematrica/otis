import AppKit
import Foundation
import UniformTypeIdentifiers
import VerbatimCore

enum SidebarSection: String, CaseIterable, Identifiable, Hashable {
    case library, search, recent, processing, completed, settings

    var id: String { rawValue }

    var title: String {
        switch self {
        case .library: return "Library"
        case .search: return "Search"
        case .recent: return "Recent Projects"
        case .processing: return "Processing"
        case .completed: return "Completed"
        case .settings: return "Settings"
        }
    }

    var systemImage: String {
        switch self {
        case .library: return "books.vertical"
        case .search: return "magnifyingglass"
        case .recent: return "clock"
        case .processing: return "waveform"
        case .completed: return "checkmark.circle"
        case .settings: return "gearshape"
        }
    }
}

struct AppAlert: Identifiable {
    let id = UUID()
    let title: String
    let message: String
}

/// A passage to show when a project opens: selected, scrolled to, cued in
/// the player, with the searched words highlighted.
struct SearchFocus: Equatable {
    let segmentID: Int
    let time: Double
    let query: String
}

/// Media staged in the import panel, validated by the engine before import.
struct StagedFile: Identifiable, Hashable {
    let id = UUID()
    let url: URL
    var media: MediaInfo?
    var error: String?
}

@MainActor
final class AppModel: ObservableObject {
    enum EngineState: Equatable {
        case starting
        case ready
        case notConfigured
        case failed(String)
    }

    @Published private(set) var engineState: EngineState = .starting
    @Published private(set) var engineInfo: EngineInfo?
    @Published private(set) var projects: [Project] = []
    @Published private(set) var jobs: [String: Job] = [:]
    @Published private(set) var modes: [TranscriptionMode] = []
    @Published private(set) var models: ModelsListing?
    @Published private(set) var downloads: [String: ModelDownload] = [:]
    @Published var alert: AppAlert?
    @Published var section: SidebarSection? = .library
    /// Ask the current section to open a project's workspace.
    @Published var openProjectRequest: String?
    /// The processing queue's paused state and order (engine-persisted).
    @Published private(set) var queue: QueueUpdate?
    /// Where to go when a project opens (from a search result), by project id.
    var pendingFocus: [String: SearchFocus] = [:]
    /// Bumped whenever a project's transcript may have changed.
    @Published private(set) var transcriptRevision: [String: Int] = [:]
    /// Latest transcript revision announced by the engine (after corrections).
    @Published private(set) var knownTranscriptRevision: [String: Int] = [:]
    /// AI review runs by id, as the engine reports their progress.
    @Published private(set) var llmRuns: [Int: LLMRun] = [:]

    let client = EngineClient()
    private var notificationTask: Task<Void, Never>?
    private var unexpectedExits = 0

    // MARK: Engine lifecycle

    func start() async {
        guard engineState != .ready else { return }
        engineState = .starting
        guard let python = EngineLocator.pythonExecutable() else {
            engineState = .notConfigured
            return
        }
        do {
            engineInfo = try await client.start(EngineLocator.configuration(python: python,
                                                                            extraEnvironment: SelfTest.engineEnvironment()))
            engineState = .ready
            listenForNotifications()
            await refreshAll()
        } catch {
            engineState = .failed(error.localizedDescription)
        }
    }

    func restart() async {
        await client.stop()
        engineState = .starting
        await start()
    }

    func shutdown() async {
        notificationTask?.cancel()
        await client.stop()
    }

    /// Stops reacting to engine events and hands back the client so the caller
    /// can stop it without involving the main actor (see AppDelegate).
    func prepareForTermination() -> EngineClient {
        notificationTask?.cancel()
        notificationTask = nil
        return client
    }

    private func listenForNotifications() {
        notificationTask?.cancel()
        let stream = client.notifications
        notificationTask = Task { [weak self] in
            for await note in stream {
                guard let self else { return }
                self.apply(note)
            }
        }
    }

    private func apply(_ note: EngineNotification) {
        switch note {
        case .jobUpdated(let job):
            let previous = jobs[job.id]
            jobs[job.id] = job
            if job.status == .completed, previous?.status != .completed {
                knownTranscriptRevision[job.projectId] = nil  // a new transcript: always reload
                transcriptRevision[job.projectId, default: 0] += 1
            }
        case .projectUpdated(let project):
            upsert(project)
        case .projectDeleted(let id):
            projects.removeAll { $0.id == id }
            jobs = jobs.filter { $0.value.projectId != id }
        case .transcriptUpdated(let update):
            knownTranscriptRevision[update.projectId] = update.revision
            transcriptRevision[update.projectId, default: 0] += 1
        case .llmRunUpdated(let run):
            llmRuns[run.id] = run
        case .queueUpdated(let update):
            queue = update
        case .modelDownload(let state):
            downloads[state.modelId] = state
            if state.isFinished {
                Task { await self.refreshModels() }
                if state.status == "failed" {
                    alert = AppAlert(title: "Download failed", message: state.error ?? "The model could not be downloaded.")
                }
            }
        case .engineTerminated(let status):
            engineState = .failed("The transcription engine stopped unexpectedly (status \(status)).")
            unexpectedExits += 1
            // Restart automatically a few times; jobs that were running are
            // marked interrupted by the engine and can be resumed.
            if unexpectedExits <= 3 {
                Task { await self.start() }
            }
        case .other:
            break
        }
    }

    /// A YouTube project was created and its first job queued.
    func noteCreated(_ created: CreatedFromYouTube) {
        upsert(created.project)
        jobs[created.job.id] = created.job
    }

    private func upsert(_ project: Project) {
        if let index = projects.firstIndex(where: { $0.id == project.id }) {
            projects[index] = project
        } else {
            projects.insert(project, at: 0)
        }
        projects.sort { ($0.updatedDate ?? .distantPast) > ($1.updatedDate ?? .distantPast) }
    }

    func refreshAll() async {
        do {
            async let projectList = client.listProjects()
            async let jobList = client.listJobs()
            async let modeList = client.modes()
            async let queueNow = client.queueState()
            projects = try await projectList
            var byID: [String: Job] = [:]
            for job in try await jobList { byID[job.id] = job }
            let state = try await queueNow
            for job in state.queued { byID[job.id] = job }
            jobs = byID
            queue = QueueUpdate(paused: state.paused, autoResume: state.autoResume, running: state.running?.id,
                                queued: state.queued.map(\.id))
            modes = try await modeList
        } catch {
            report(error, title: "Could not load projects")
        }
        await refreshModels()
    }

    func refreshModels() async {
        do {
            let listing = try await client.models()
            models = listing
            for (id, state) in listing.downloads { downloads[id] = state }
        } catch {
            report(error, title: "Could not load models")
        }
    }

    // MARK: Derived state

    func latestJob(for projectID: String) -> Job? {
        jobs.values.filter { $0.projectId == projectID }.max { $0.createdAt < $1.createdAt }
    }

    /// The running job, then the queued ones in the order they will run.
    var activeJobs: [Job] {
        let order = Dictionary(uniqueKeysWithValues: (queue?.queued ?? []).enumerated().map { ($1, $0) })
        return jobs.values.filter { $0.status.isActive }.sorted {
            let a = $0.status == .running ? -1 : (order[$0.id] ?? $0.queuePosition ?? Int.max)
            let b = $1.status == .running ? -1 : (order[$1.id] ?? $1.queuePosition ?? Int.max)
            return (a, $0.createdAt) < (b, $1.createdAt)
        }
    }

    var queuedJobs: [Job] { activeJobs.filter { $0.status == .queued } }

    // MARK: Queue

    func pauseQueue(interrupt: Bool) async {
        do { apply(try await client.pauseQueue(interrupt: interrupt)) } catch { report(error, title: "Could not pause") }
    }

    func resumeQueue() async {
        do { apply(try await client.resumeQueue()) } catch { report(error, title: "Could not resume the queue") }
    }

    /// Moves a queued job up (-1) or down (+1).
    func move(_ job: Job, by offset: Int) async {
        var order = queuedJobs.map(\.id)
        guard let index = order.firstIndex(of: job.id) else { return }
        let target = index + offset
        guard order.indices.contains(target) else { return }
        order.swapAt(index, target)
        do { apply(try await client.reorderQueue(order)) } catch { report(error, title: "Could not reorder the queue") }
    }

    func setAutoResume(_ enabled: Bool) async {
        do { apply(try await client.setAutoResume(enabled)) } catch { report(error, title: "Could not change the setting") }
    }

    private func apply(_ state: QueueState) {
        for job in state.queued { jobs[job.id] = job }
        if let running = state.running { jobs[running.id] = running }
        queue = QueueUpdate(paused: state.paused, autoResume: state.autoResume, running: state.running?.id,
                            queued: state.queued.map(\.id))
    }

    var recentlyFinishedJobs: [Job] {
        jobs.values.filter { !$0.status.isActive }.sorted { $0.updatedAt > $1.updatedAt }.prefix(20).map { $0 }
    }

    func project(_ id: String) -> Project? { projects.first { $0.id == id } }

    func modelEntry(_ id: String) -> ModelEntry? { models?.models.first { $0.id == id } }

    func installedModels(kind: String) -> [ModelEntry] {
        models?.models.filter { $0.modelKind == kind && $0.status.installed } ?? []
    }

    // MARK: Import & jobs

    func stage(_ urls: [URL]) async -> [StagedFile] {
        var staged: [StagedFile] = []
        for url in urls {
            var file = StagedFile(url: url)
            do {
                file.media = try await client.probe(path: url.path)
            } catch {
                file.error = error.localizedDescription
            }
            staged.append(file)
        }
        return staged
    }

    func preflight(_ options: JobOptions) async -> Result<Preflight, EngineClientError> {
        do {
            return .success(try await client.preflight(options))
        } catch let error as EngineClientError {
            return .failure(error)
        } catch {
            return .failure(.invalidResponse(error.localizedDescription))
        }
    }

    /// Creates a project for each valid file and queues its transcription.
    func transcribe(_ files: [StagedFile], options: JobOptions) async -> Bool {
        var started = false
        for file in files where file.error == nil {
            do {
                let project = try await client.createProject(path: file.url.path)
                upsert(project)
                let job = try await client.startJob(projectId: project.id, options: options)
                jobs[job.id] = job
                started = true
            } catch {
                report(error, title: "Could not start \(file.url.lastPathComponent)")
            }
        }
        return started
    }

    func retranscribe(_ project: Project, options: JobOptions) async {
        do {
            let job = try await client.startJob(projectId: project.id, options: options)
            jobs[job.id] = job
        } catch {
            report(error, title: "Could not start transcription")
        }
    }

    func cancel(_ job: Job) async {
        do { jobs[job.id] = try await client.cancelJob(job.id) } catch { report(error, title: "Could not cancel") }
    }

    func resume(_ job: Job) async {
        do { jobs[job.id] = try await client.resumeJob(job.id) } catch { report(error, title: "Could not resume") }
    }

    func delete(_ project: Project) async {
        do {
            try await client.deleteProject(project.id)
            projects.removeAll { $0.id == project.id }
            jobs = jobs.filter { $0.value.projectId != project.id }
        } catch {
            report(error, title: "Could not delete project")
        }
    }

    func rename(_ project: Project, to title: String) async {
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        do { upsert(try await client.renameProject(project.id, title: trimmed)) } catch { report(error, title: "Could not rename") }
    }

    /// The transcript with its words, as the workspace loads it.
    func transcript(for project: Project) async throws -> Transcript {
        try await client.transcript(projectId: project.id, includeWords: true)
    }

    /// Saves a caption track exactly as it was published (not the transcript).
    func exportCaptions(_ project: Project, track: CaptionInfo, format: String) async {
        do {
            let preview = try await client.renderCaptions(projectId: project.id, captionId: track.id, format: format)
            let panel = NSSavePanel()
            panel.nameFieldStringValue = preview.suggestedFilename
            panel.allowedContentTypes = [UTType(filenameExtension: format) ?? .plainText]
            panel.canCreateDirectories = true
            guard panel.runModal() == .OK, let url = panel.url else { return }
            _ = try await client.writeCaptions(projectId: project.id, captionId: track.id, format: format, to: url.path)
        } catch {
            report(error, title: "Export failed")
        }
    }

    /// Asks where to save, then has the engine write the file (atomically).
    func export(_ project: Project, format: ExportFormat, options: ExportOptions = .init()) async {
        do {
            let preview = try await client.renderExport(projectId: project.id, format: format.rawValue, options: options)
            let panel = NSSavePanel()
            panel.nameFieldStringValue = preview.suggestedFilename
            panel.allowedContentTypes = [UTType(filenameExtension: format.fileExtension) ?? .plainText]
            panel.canCreateDirectories = true
            guard panel.runModal() == .OK, let url = panel.url else { return }
            _ = try await client.writeExport(projectId: project.id, to: url.path, format: format.rawValue, options: options)
        } catch {
            report(error, title: "Export failed")
        }
    }

    // MARK: Models

    func download(_ model: ModelEntry) async {
        do { downloads[model.id] = try await client.downloadModel(model.id) } catch { report(error, title: "Download failed") }
    }

    func cancelDownload(_ model: ModelEntry) async {
        do { try await client.cancelDownload(model.id) } catch { report(error, title: "Could not cancel download") }
    }

    func deleteModel(_ model: ModelEntry) async {
        do {
            try await client.deleteModel(model.id)
            await refreshModels()
        } catch {
            report(error, title: "Could not delete model")
        }
    }

    func setModelsDirectory(_ url: URL?) async {
        do {
            _ = try await client.setModelsDirectory(url?.path)
            await refreshModels()
        } catch {
            report(error, title: "Could not change the models folder")
        }
    }

    // MARK: Errors

    /// Records a review run returned by a call (later updates arrive as notifications).
    func noteLLMRun(_ run: LLMRun) {
        llmRuns[run.id] = run
    }

    func report(_ error: Error, title: String) {
        alert = AppAlert(title: title, message: error.localizedDescription)
    }
}
