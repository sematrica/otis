import SwiftUI
import VerbatimCore

/// A centred placeholder (ContentUnavailableView requires macOS 14).
struct EmptyState<Actions: View>: View {
    let title: String
    let systemImage: String
    let message: String?
    @ViewBuilder var actions: () -> Actions

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: systemImage)
                .font(.system(size: 40, weight: .light))
                .foregroundStyle(.secondary)
            Text(title).font(.title3.weight(.semibold))
            if let message {
                Text(message)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 460)
                    .textSelection(.enabled)
            }
            actions()
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

extension EmptyState where Actions == EmptyView {
    init(title: String, systemImage: String, message: String?) {
        self.init(title: title, systemImage: systemImage, message: message) { EmptyView() }
    }
}

struct StatusBadge: View {
    let status: ProjectStatus

    var body: some View {
        Text(label)
            .font(.caption.weight(.medium))
            .padding(.horizontal, 7)
            .padding(.vertical, 2)
            .background(color.opacity(0.15), in: Capsule())
            .foregroundStyle(color)
    }

    private var label: String {
        switch status {
        case .new: return "New"
        case .queued: return "Queued"
        case .processing: return "Processing"
        case .ready: return "Ready"
        case .failed: return "Failed"
        case .cancelled: return "Cancelled"
        case .interrupted: return "Interrupted"
        }
    }

    private var color: Color {
        switch status {
        case .ready: return .green
        case .processing, .queued: return .blue
        case .failed: return .red
        case .cancelled, .interrupted: return .orange
        case .new: return .secondary
        }
    }
}

struct ProjectRow: View {
    let project: Project

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: project.hasVideo ? "film" : "waveform")
                .font(.title3)
                .foregroundStyle(.secondary)
                .frame(width: 26)
            VStack(alignment: .leading, spacing: 2) {
                Text(project.title).font(.body.weight(.medium)).lineLimit(1)
                HStack(spacing: 6) {
                    Text(Timecode.duration(project.duration))
                    if let language = project.language { Text("· \(language)") }
                    if project.segmentCount > 0 { Text("· \(project.segmentCount) segments") }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            Spacer()
            StatusBadge(status: project.status)
        }
        .padding(.vertical, 4)
    }
}

struct ProjectListView: View {
    let title: String
    let projects: [Project]
    let onOpen: (String) -> Void

    var body: some View {
        Group {
            if projects.isEmpty {
                EmptyState(title: "No projects", systemImage: "tray", message: "Import a recording from the Library.")
            } else {
                List(projects) { project in
                    Button { onOpen(project.id) } label: { ProjectRow(project: project) }
                        .buttonStyle(.plain)
                }
            }
        }
        .navigationTitle(title)
    }
}
