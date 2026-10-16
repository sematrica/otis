import AppKit
import SwiftUI
import VerbatimCore

struct SettingsView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        TabView {
            ModelsSettings()
                .tabItem { Label("Models", systemImage: "cpu") }
            EngineSettingsView()
                .tabItem { Label("Engine", systemImage: "gearshape.2") }
        }
        .padding(16)
        .navigationTitle("Settings")
    }
}

private struct ModelsSettings: View {
    @EnvironmentObject private var model: AppModel
    @State private var pendingDownload: ModelEntry?
    @State private var pendingDelete: ModelEntry?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            folderRow
            if let listing = model.models {
                List {
                    ForEach(["mlx", "faster-whisper"], id: \.self) { backend in
                        Section(header: sectionHeader(backend, listing: listing)) {
                            ForEach(listing.models.filter { $0.backend == backend }) { entry in
                                ModelRow(entry: entry, download: model.downloads[entry.id],
                                         onDownload: { pendingDownload = entry },
                                         onCancel: { Task { await model.cancelDownload(entry) } },
                                         onDelete: { pendingDelete = entry })
                            }
                        }
                    }
                }
                .listStyle(.inset(alternatesRowBackgrounds: true))
            } else {
                ProgressView()
            }
            Text("Sizes are approximate. Memory figures are estimates for one loaded model and have not been measured on every Mac. "
                 + "Models are only downloaded when you confirm.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .task { await model.refreshModels() }
        .confirmationDialog(
            pendingDownload.map { "Download \($0.displayName)?" } ?? "",
            isPresented: Binding(get: { pendingDownload != nil }, set: { if !$0 { pendingDownload = nil } }),
            presenting: pendingDownload
        ) { entry in
            Button("Download \(ByteSize.string(entry.approxDownloadBytes))") { Task { await model.download(entry) } }
            Button("Cancel", role: .cancel) {}
        } message: { entry in
            Text("About \(ByteSize.string(entry.approxDownloadBytes)) from \(entry.sourceDescription) "
                 + "will be stored in \(model.models?.modelsDir ?? "the models folder").")
        }
        .confirmationDialog(
            pendingDelete.map { "Delete \($0.displayName) (\($0.backend))?" } ?? "",
            isPresented: Binding(get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } }),
            presenting: pendingDelete
        ) { entry in
            Button("Delete", role: .destructive) { Task { await model.deleteModel(entry) } }
            Button("Cancel", role: .cancel) {}
        } message: { entry in
            Text("Frees \(ByteSize.string(entry.status.sizeOnDisk)). You can download it again later.")
        }
    }

    private var folderRow: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Models folder").font(.headline)
                Text(model.models?.modelsDir ?? "—")
                    .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            }
            Spacer()
            Button("Show in Finder") {
                if let dir = model.models?.modelsDir {
                    NSWorkspace.shared.open(URL(fileURLWithPath: dir, isDirectory: true))
                }
            }
            Button("Change…") { chooseFolder() }
            Button("Use Default") { Task { await model.setModelsDirectory(nil) } }
        }
    }

    private func sectionHeader(_ backend: String, listing: ModelsListing) -> some View {
        let info = listing.backends.first { $0.name == backend }
        return HStack {
            Text(info?.displayName ?? backend)
            if let info, !info.available {
                Text("— unavailable: \(info.unavailableReason ?? "unknown reason")").foregroundStyle(.orange)
            } else if let info, !info.capabilities.beamSearch {
                Text("— greedy decoding only").foregroundStyle(.secondary)
            }
        }
    }

    private func chooseFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.prompt = "Use Folder"
        panel.message = "Choose where speech models are stored (for example, on an external SSD). Existing models are not moved."
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task { await model.setModelsDirectory(url) }
    }
}

private struct ModelRow: View {
    let entry: ModelEntry
    let download: ModelDownload?
    let onDownload: () -> Void
    let onCancel: () -> Void
    let onDelete: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(entry.displayName).font(.body.weight(.medium))
                Text(details).font(.caption).foregroundStyle(.secondary)
                if !entry.notes.isEmpty {
                    Text(entry.notes).font(.caption).foregroundStyle(.secondary)
                }
            }
            Spacer()
            if let download, !download.isFinished {
                DownloadProgress(state: download)
                Button("Cancel", action: onCancel).controlSize(.small)
            } else if entry.status.installed {
                Label("Installed", systemImage: "checkmark.circle.fill").foregroundStyle(.green).font(.caption)
                Button("Delete…", action: onDelete).controlSize(.small)
            } else {
                Button("Download…", action: onDownload).controlSize(.small)
            }
        }
        .padding(.vertical, 2)
    }

    private var details: String {
        let memory = entry.memoryEstimates
            .sorted { $0.key < $1.key }
            .map { "\($0.key): ~\(ByteSize.string($0.value)) RAM" }
            .joined(separator: ", ")
        let size = entry.status.installed
            ? "\(ByteSize.string(entry.status.sizeOnDisk)) on disk"
            : "~\(ByteSize.string(entry.approxDownloadBytes)) download"
        return "\(entry.parameters) parameters · \(size) · \(memory)"
    }
}

private struct EngineSettingsView: View {
    @EnvironmentObject private var model: AppModel
    @AppStorage(EngineLocator.defaultsKey) private var pythonOverride = ""

    var body: some View {
        Form {
            if let info = model.engineInfo {
                Section("Engine") {
                    LabeledContent("Version", value: "\(info.version) (protocol \(info.protocolVersion))")
                    LabeledContent("Python", value: "\(info.python.version) — \(info.python.executable)")
                    LabeledContent("Data folder", value: info.dataDir)
                    LabeledContent("Hardware", value: hardware(info.hardware))
                }
                Section("Speech-recognition backends") {
                    ForEach(info.backends) { backend in
                        LabeledContent(backend.displayName) {
                            Text(backend.available
                                 ? "\(backend.capabilities.acceleration); beam search: \(backend.capabilities.beamSearch ? "yes" : "no")"
                                 : (backend.unavailableReason ?? "unavailable"))
                                .foregroundStyle(backend.available ? .primary : .secondary)
                        }
                    }
                }
            }
            Section("Advanced") {
                TextField("Python runtime override", text: $pythonOverride, prompt: Text("Use the embedded runtime"))
                Text("For development only: path to a Python executable with verbatim-engine installed. Takes effect after restarting the engine.")
                    .font(.caption).foregroundStyle(.secondary)
                HStack {
                    Button("Restart Engine") { Task { await model.restart() } }
                    Button("Open Logs Folder") {
                        let logs = EngineLocator.dataDirectory().appendingPathComponent("logs", isDirectory: true)
                        NSWorkspace.shared.open(logs)
                    }
                }
            }
        }
        .formStyle(.grouped)
    }

    private func hardware(_ hw: EngineInfo.Hardware) -> String {
        var parts = [hw.machine]
        if let os = hw.macosVersion { parts.append("macOS \(os)") }
        parts.append("\(hw.performanceCores) performance cores")
        if let memory = hw.memoryBytes { parts.append("\(ByteSize.string(memory)) memory") }
        return parts.joined(separator: " · ")
    }
}
