import SwiftUI
import VerbatimCore

/// Full-text search over every transcript in the library. Results are
/// grouped by project; choosing one opens the project at that passage.
struct SearchView: View {
    let onOpen: (String) -> Void
    @EnvironmentObject private var model: AppModel
    @State private var query = ""
    @State private var includeOriginal = false
    @State private var response: SearchResponse?
    @State private var error: String?
    @State private var isSearching = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                    TextField("Search all transcripts — words, or a \"phrase\"", text: $query)
                        .textFieldStyle(.plain)
                        .font(.title3)
                        .onSubmit { Task { await run() } }
                    if isSearching { ProgressView().controlSize(.small) }
                }
                Toggle("Also search the AI's wording of passages you corrected", isOn: $includeOriginal)
                    .font(.caption)
                    .toggleStyle(.checkbox)
            }
            .padding(14)
            Divider()
            results
        }
        .navigationTitle("Search")
        // search as you type, once the typing pauses
        .task(id: "\(query)\u{1}\(includeOriginal)") {
            try? await Task.sleep(for: .milliseconds(300))
            guard !Task.isCancelled else { return }
            await run()
        }
    }

    @ViewBuilder
    private var results: some View {
        if let error {
            EmptyState(title: "Search failed", systemImage: "exclamationmark.triangle", message: error) { EmptyView() }
        } else if let response, !response.results.isEmpty {
            List {
                ForEach(groups(response.results)) { group in
                    Section {
                        ForEach(group.hits) { hit in
                            Button { open(hit) } label: { HitRow(hit: hit) }
                                .buttonStyle(.plain)
                        }
                    } header: {
                        Text("\(group.title) — \(group.hits.count) passage\(group.hits.count == 1 ? "" : "s")")
                    }
                }
                if response.truncated {
                    Text("Only the first \(response.results.count) passages are shown. Add words to narrow the search.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .listStyle(.inset)
        } else if response != nil {
            EmptyState(title: "Nothing found", systemImage: "text.magnifyingglass",
                       message: "No transcript contains all of these words.") { EmptyView() }
        } else {
            EmptyState(title: "Search your transcripts", systemImage: "magnifyingglass",
                       message: "Words are matched by their beginning, ignoring case and accents: “bridg” finds "
                           + "“bridge” and “Bridges”. Put several words in quotes to find them together.") { EmptyView() }
        }
    }

    private struct ProjectHits: Identifiable {
        let id: String
        let title: String
        let hits: [SearchHit]
    }

    /// Projects in the order of their best match; passages in time order.
    private func groups(_ hits: [SearchHit]) -> [ProjectHits] {
        var order: [String] = []
        var byProject: [String: [SearchHit]] = [:]
        for hit in hits {
            if byProject[hit.projectId] == nil { order.append(hit.projectId) }
            byProject[hit.projectId, default: []].append(hit)
        }
        return order.map { id in
            let found = byProject[id] ?? []
            return ProjectHits(id: id, title: found.first?.projectTitle ?? "", hits: found.sorted { $0.start < $1.start })
        }
    }

    private func run() async {
        let text = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            response = nil
            error = nil
            return
        }
        isSearching = true
        defer { isSearching = false }
        do {
            response = try await model.client.searchTranscripts(text, includeOriginal: includeOriginal, limit: 300)
            error = nil
        } catch let failure as EngineClientError where failure.engineCode == "invalid_params" {
            response = nil
            error = nil  // only punctuation typed so far
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func open(_ hit: SearchHit) {
        model.pendingFocus[hit.projectId] = SearchFocus(segmentID: hit.segmentId, time: hit.start, query: query)
        onOpen(hit.projectId)
    }
}

private struct HitRow: View {
    let hit: SearchHit

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Text(Timecode.clock(hit.start))
                .font(.callout.monospacedDigit())
                .foregroundStyle(Color.accentColor)
                .frame(width: 70, alignment: .leading)
            VStack(alignment: .leading, spacing: 2) {
                if let speaker = hit.speaker {
                    Text(speaker).font(.caption.weight(.medium)).foregroundStyle(.secondary)
                }
                Text(highlighted).lineLimit(3)
                if hit.inOriginalOnly {
                    Text("Found in the AI's wording before your correction")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
        }
        .padding(.vertical, 3)
        .contentShape(Rectangle())
    }

    private var highlighted: AttributedString {
        var out = AttributedString()
        for (piece, marked) in hit.runs {
            var run = AttributedString(piece)
            if marked {
                run[AttributeScopes.SwiftUIAttributes.BackgroundColorAttribute.self] = Color.yellow.opacity(0.45)
                run[AttributeScopes.SwiftUIAttributes.FontAttribute.self] = .body.weight(.semibold)
            }
            out += run
        }
        return out
    }
}
