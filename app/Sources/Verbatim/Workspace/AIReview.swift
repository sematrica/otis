import SwiftUI
import VerbatimCore

/// Settings for optional AI verification (Settings › AI Verification). The
/// OpenAI API key is kept in the macOS Keychain, never in preferences.
enum AISettings {
    static let providerKey = "ai.provider"          // "off" | "ollama" | "openai"
    static let ollamaURLKey = "ai.ollama.url"
    static let ollamaModelKey = "ai.ollama.model"
    static let openAIModelKey = "ai.openai.model"
    static let allowCloudKey = "ai.openai.allowed"  // the user enabled sending text to OpenAI
    static let keychainAccount = "openai-api-key"
    static let defaultOllamaURL = "http://127.0.0.1:11434"

    static var provider: String { UserDefaults.standard.string(forKey: providerKey) ?? "off" }
    static var ollamaURL: String {
        let value = UserDefaults.standard.string(forKey: ollamaURLKey) ?? ""
        return value.isEmpty ? defaultOllamaURL : value
    }
    static var model: String {
        UserDefaults.standard.string(forKey: provider == "openai" ? openAIModelKey : ollamaModelKey) ?? ""
    }
    static var cloudAllowed: Bool { UserDefaults.standard.bool(forKey: allowCloudKey) }
    static var providerName: String { provider == "openai" ? "OpenAI" : "Ollama" }
    /// Where the transcript text would go.
    static var endpointDescription: String {
        provider == "openai" ? "OpenAI (api.openai.com, a cloud service)" : "Ollama at \(ollamaURL)"
    }
    static var isRemote: Bool {
        if provider == "openai" { return true }
        let host = URL(string: ollamaURL)?.host?.lowercased() ?? ""
        return !(host == "localhost" || host == "127.0.0.1" || host == "::1")
    }
}

/// One AI review of the open transcript and what came of it.
@MainActor
final class AIReviewModel: ObservableObject {
    @Published private(set) var runID: Int?
    @Published private(set) var suggestions: [Suggestion] = []
    @Published private(set) var error: String?
    @Published private(set) var isStarting = false
    @Published private(set) var busySuggestion: Int?

    let projectID: String
    private let app: AppModel
    private weak var workspace: WorkspaceModel?

    init(projectID: String, app: AppModel, workspace: WorkspaceModel) {
        self.projectID = projectID
        self.app = app
        self.workspace = workspace
    }

    var run: LLMRun? { runID.flatMap { app.llmRuns[$0] } }
    var pending: [Suggestion] { suggestions.filter { $0.status == "pending" } }
    var decided: [Suggestion] { suggestions.filter { ["accepted", "rejected", "outdated"].contains($0.status) } }
    var invalid: [Suggestion] { suggestions.filter { $0.status == "invalid" } }

    func load() async {
        do {
            suggestions = try await app.client.suggestions(projectId: projectID)
            if runID == nil, let latest = try await app.client.llmRuns(projectId: projectID).first {
                runID = latest.id
                app.noteLLMRun(latest)
            }
        } catch {
            self.error = error.localizedDescription
        }
    }

    /// Starts a review with the provider from Settings. `confirmedRemote` is
    /// true only after the user agreed to send the text off this computer.
    func start(confirmedRemote: Bool) async {
        error = nil
        isStarting = true
        defer { isStarting = false }
        let provider = AISettings.provider
        let model = AISettings.model
        let key = provider == "openai" ? KeychainStore.read(account: AISettings.keychainAccount) : nil
        do {
            let run = try await app.client.llmReview(
                projectId: projectID, provider: provider, model: model,
                baseURL: provider == "ollama" ? AISettings.ollamaURL : nil, apiKey: key, allowRemote: confirmedRemote)
            runID = run.id
            app.noteLLMRun(run)
        } catch {
            self.error = error.localizedDescription
        }
    }

    func cancel() async {
        guard let runID else { return }
        _ = try? await app.client.cancelLLMRun(id: runID)
    }

    func accept(_ suggestion: Suggestion) async {
        busySuggestion = suggestion.id
        defer { busySuggestion = nil }
        do {
            let result = try await app.client.acceptSuggestion(id: suggestion.id)
            if result.suggestion.status == "outdated" {
                error = "The passage changed since the review, so this suggestion no longer applies."
            }
            if result.transcript != nil { await workspace?.load() }
            await load()
        } catch {
            app.report(error, title: "Could not apply the suggestion")
        }
    }

    func reject(_ suggestion: Suggestion) async {
        busySuggestion = suggestion.id
        defer { busySuggestion = nil }
        do {
            _ = try await app.client.rejectSuggestion(id: suggestion.id)
            await load()
        } catch {
            app.report(error, title: "Could not reject the suggestion")
        }
    }
}

struct AIReviewSheet: View {
    @StateObject private var review: AIReviewModel
    @ObservedObject var workspace: WorkspaceModel
    let onSeek: (Double) -> Void
    @EnvironmentObject private var app: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var confirmRemote = false

    init(projectID: String, app: AppModel, workspace: WorkspaceModel, onSeek: @escaping (Double) -> Void) {
        _review = StateObject(wrappedValue: AIReviewModel(projectID: projectID, app: app, workspace: workspace))
        self.workspace = workspace
        self.onSeek = onSeek
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Review with AI").font(.title3.weight(.semibold))
            Text("A language model reads the transcript text and suggests passages to check, punctuation, "
                 + "inconsistent spellings and passages to re-check against the audio. It cannot hear the recording: "
                 + "nothing is changed until you accept a suggestion, and every accepted change can be undone.")
                .font(.callout).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            providerBox
            if let error = review.error {
                Label(error, systemImage: "exclamationmark.triangle").font(.callout).foregroundStyle(.orange)
            }
            Divider()
            suggestionList
            HStack {
                if let run = review.run {
                    RunStatus(run: run)
                }
                Spacer()
                Button("Close") { dismiss() }.keyboardShortcut(.cancelAction)
            }
        }
        .padding(18)
        .frame(width: 640, height: 600)
        .task { await review.load() }
        .onChange(of: review.run?.status) { _ in Task { await review.load() } }
        .onChange(of: review.run?.requests) { _ in Task { await review.load() } }
        .confirmationDialog("Send the transcript text to \(AISettings.providerName)?", isPresented: $confirmRemote,
                            titleVisibility: .visible) {
            Button("Send Text and Review") { Task { await review.start(confirmedRemote: true) } }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("The text of this transcript (\(wordCount) words) will be sent to \(AISettings.endpointDescription). "
                 + "The audio is never sent.")
        }
    }

    private var wordCount: Int {
        workspace.segments.reduce(0) { $0 + $1.text.split(separator: " ").count }
    }

    @ViewBuilder
    private var providerBox: some View {
        let provider = AISettings.provider
        GroupBox {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    if provider == "off" {
                        Text("AI verification is off.").font(.callout)
                        Text("Choose a provider in Settings › AI Verification.").font(.caption).foregroundStyle(.secondary)
                    } else {
                        Text("\(AISettings.providerName) · \(AISettings.model.isEmpty ? "no model chosen" : AISettings.model)")
                            .font(.callout.weight(.medium))
                        Text(AISettings.isRemote ? "Text is sent to \(AISettings.endpointDescription) after you confirm."
                                                 : "Runs on this computer; nothing leaves it.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                Spacer()
                if review.run?.isRunning == true {
                    Button("Stop") { Task { await review.cancel() } }
                } else {
                    Button("Start Review") {
                        if AISettings.isRemote {
                            confirmRemote = true
                        } else {
                            Task { await review.start(confirmedRemote: false) }
                        }
                    }
                    .disabled(provider == "off" || AISettings.model.isEmpty || review.isStarting
                              || (provider == "openai" && !AISettings.cloudAllowed))
                    .help(provider == "openai" && !AISettings.cloudAllowed
                          ? "Allow sending transcript text to OpenAI in Settings › AI Verification first" : "")
                }
            }
            .padding(4)
        }
    }

    @ViewBuilder
    private var suggestionList: some View {
        if review.suggestions.isEmpty {
            EmptyState(title: "No suggestions yet", systemImage: "sparkles",
                       message: review.run?.isRunning == true ? "The review is running." : "Start a review to get suggestions.")
        } else {
            List {
                if !review.pending.isEmpty {
                    Section("To decide (\(review.pending.count))") {
                        ForEach(review.pending) { suggestion in
                            SuggestionRow(suggestion: suggestion, segment: workspace.segment(suggestion.segmentId),
                                          busy: review.busySuggestion == suggestion.id, onSeek: onSeek,
                                          onAccept: { Task { await review.accept(suggestion) } },
                                          onReject: { Task { await review.reject(suggestion) } })
                        }
                    }
                }
                if !review.decided.isEmpty {
                    Section("Decided (\(review.decided.count))") {
                        ForEach(review.decided) { suggestion in
                            SuggestionRow(suggestion: suggestion, segment: workspace.segment(suggestion.segmentId),
                                          busy: false, onSeek: onSeek, onAccept: nil, onReject: nil)
                        }
                    }
                }
                if !review.invalid.isEmpty {
                    Section("Discarded by validation (\(review.invalid.count))") {
                        ForEach(review.invalid) { suggestion in
                            SuggestionRow(suggestion: suggestion, segment: workspace.segment(suggestion.segmentId),
                                          busy: false, onSeek: onSeek, onAccept: nil, onReject: nil)
                        }
                    }
                }
            }
            .listStyle(.inset)
        }
    }
}

private struct RunStatus: View {
    let run: LLMRun

    var body: some View {
        HStack(spacing: 6) {
            if run.isRunning { ProgressView().controlSize(.small) }
            Text(text).font(.caption).foregroundStyle(run.status == "failed" ? Color.orange : Color.secondary)
        }
        .help("\(run.provider) · \(run.model) at \(run.endpoint)")
    }

    private var text: String {
        switch run.status {
        case "running": return "Reviewing… \(run.segmentsSent) passages sent in \(run.requests) request(s)"
        case "completed": return "Review finished: \(run.segmentsSent) passages, \(run.requests) request(s)"
        case "cancelled": return "Review stopped"
        default: return "Review failed: \(run.error ?? "unknown error")"
        }
    }
}

private struct SuggestionRow: View {
    let suggestion: Suggestion
    let segment: Segment?
    let busy: Bool
    let onSeek: (Double) -> Void
    let onAccept: (() -> Void)?
    let onReject: (() -> Void)?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Text(kindLabel).font(.caption.weight(.semibold))
                    .padding(.horizontal, 6).padding(.vertical, 1)
                    .background(Color.secondary.opacity(0.15), in: Capsule())
                if let segment {
                    Button(Timecode.clock(segment.start)) { onSeek(segment.start) }
                        .buttonStyle(.link).font(.caption.monospacedDigit())
                }
                Spacer()
                if suggestion.status != "pending" {
                    Text(suggestion.status.capitalized).font(.caption).foregroundStyle(.secondary)
                }
                if busy { ProgressView().controlSize(.small) }
                if let onReject { Button("Reject", action: onReject).disabled(busy) }
                if let onAccept { Button(acceptLabel, action: onAccept).disabled(busy) }
            }
            Text(suggestion.originalText).font(.callout).foregroundStyle(.secondary).textSelection(.enabled)
            if let suggested = suggestion.suggestedText {
                Text("→ \(suggested)").font(.callout).textSelection(.enabled)
            }
            if let reason = suggestion.reason, !reason.isEmpty {
                Text(reason).font(.caption).foregroundStyle(.secondary)
            }
            if let validation = suggestion.validation {
                Text("Not offered: \(validation)").font(.caption).foregroundStyle(.orange)
            }
        }
        .padding(.vertical, 3)
    }

    private var kindLabel: String {
        switch suggestion.kind {
        case "punctuation": return "Punctuation"
        case "terminology": return "Spelling"
        case "retranscribe": return "Re-check audio"
        default: return "Check passage"
        }
    }

    private var acceptLabel: String {
        switch suggestion.kind {
        case "suspicious": return "Mark Uncertain"
        case "retranscribe": return "Re-check"
        default: return "Apply"
        }
    }
}
