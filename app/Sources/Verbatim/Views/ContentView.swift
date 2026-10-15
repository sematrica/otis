import AppKit
import SwiftUI
import VerbatimCore

struct ContentView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        NavigationSplitView {
            List(SidebarSection.allCases, selection: $model.section) { section in
                Label(section.title, systemImage: section.systemImage)
                    .badge(section == .processing ? model.activeJobs.count : 0)
                    .tag(section)
            }
            .navigationSplitViewColumnWidth(min: 180, ideal: 200)
        } detail: {
            switch model.engineState {
            case .ready:
                SectionView(section: model.section ?? .library)
            case .starting:
                EmptyState(title: "Starting transcription engine…", systemImage: "hourglass", message: nil) {
                    ProgressView().controlSize(.small)
                }
            case .notConfigured:
                EmptyState(
                    title: "Transcription engine not found",
                    systemImage: "exclamationmark.triangle",
                    message: "This build has no embedded engine. Build the app with scripts/build_app.sh, "
                        + "or set the Python runtime in Settings › Advanced."
                ) {
                    OpenSettingsButton()
                }
            case .failed(let message):
                EmptyState(title: "The transcription engine is not running", systemImage: "exclamationmark.octagon", message: message) {
                    Button("Restart Engine") { Task { await model.restart() } }
                }
            }
        }
        .alert(item: $model.alert) { alert in
            Alert(title: Text(alert.title), message: Text(alert.message))
        }
    }
}

/// Opens the Settings window (SwiftUI's SettingsLink requires macOS 14).
struct OpenSettingsButton: View {
    var body: some View {
        Button("Open Settings…") {
            NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
        }
    }
}

private struct SectionView: View {
    let section: SidebarSection
    @EnvironmentObject private var model: AppModel
    @State private var path = NavigationPath()

    var body: some View {
        NavigationStack(path: $path) {
            Group {
                switch section {
                case .library:
                    LibraryView(onOpen: open)
                case .recent:
                    ProjectListView(title: "Recent Projects", projects: Array(model.projects.prefix(15)), onOpen: open)
                case .processing:
                    ProcessingView(onOpen: open)
                case .completed:
                    ProjectListView(title: "Completed", projects: model.projects.filter { $0.status == .ready }, onOpen: open)
                case .settings:
                    SettingsView()
                }
            }
            .navigationDestination(for: String.self) { projectID in
                ProjectWorkspaceView(projectID: projectID)
            }
        }
        .id(section)  // reset navigation when switching sections
    }

    private func open(_ projectID: String) {
        path.append(projectID)
    }
}
