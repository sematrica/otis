import SwiftUI
import VerbatimCore

/// Left panel: the speakers found in the recording and their management
/// (rename, merge, add, show one speaker's segments). Speakers are anonymous
/// until the user names them.
struct SpeakersPanel: View {
    @ObservedObject var workspace: WorkspaceModel
    @ObservedObject var player: PlayerController
    @State private var renaming: Speaker?
    @State private var addingSpeaker = false
    @State private var merging: (source: Speaker, target: Speaker)?

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Speakers").font(.headline)
                Spacer()
                Button {
                    addingSpeaker = true
                } label: { Image(systemName: "plus") }
                    .buttonStyle(.borderless)
                    .help("Add a speaker (for example, to split one speaker into two)")
                    .disabled(workspace.transcript == nil)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            Divider()
            if workspace.speakers.isEmpty {
                noSpeakers
            } else {
                List(selection: Binding(get: { workspace.focusedSpeakerID }, set: select)) {
                    ForEach(workspace.speakers) { speaker in
                        SpeakerRow(speaker: speaker, isSpeaking: isSpeaking(speaker))
                            .tag(speaker.id)
                            .contextMenu { menu(for: speaker) }
                    }
                }
                .listStyle(.sidebar)
            }
            Divider()
            footer
        }
        .sheet(item: $renaming) { speaker in
            NameSheet(title: "Rename \(speaker.label)",
                      prompt: "Name shown for this speaker (leave empty to use “\(speaker.label)”):",
                      initial: speaker.name ?? "", action: "Rename") { name in
                Task { await workspace.rename(speaker, to: name) }
            }
        }
        .sheet(isPresented: $addingSpeaker) {
            NameSheet(title: "Add Speaker", prompt: "Name (optional). Assign segments to the new speaker from the transcript.",
                      initial: "", action: "Add") { name in
                Task { await workspace.addSpeaker(named: name) }
            }
        }
        .confirmationDialog(
            merging.map { "Merge “\($0.source.displayName)” into “\($0.target.displayName)”?" } ?? "",
            isPresented: Binding(get: { merging != nil }, set: { if !$0 { merging = nil } }),
            titleVisibility: .visible
        ) {
            Button("Merge") {
                if let pair = merging { Task { await workspace.merge(pair.source, into: pair.target) } }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("All of \(merging?.source.displayName ?? "")'s segments are assigned to \(merging?.target.displayName ?? ""). "
                 + "You can undo this.")
        }
    }

    private func select(_ id: Int?) {
        workspace.focusedSpeakerID = id
        if id != nil, workspace.filter == .all { workspace.filter = .speaker }
        if id == nil, workspace.filter == .speaker { workspace.filter = .all }
    }

    private func isSpeaking(_ speaker: Speaker) -> Bool {
        guard player.isPlaying, let id = workspace.activeSegmentID(at: player.currentTime) else { return false }
        return workspace.segment(id)?.speakerId == speaker.id
    }

    @ViewBuilder
    private func menu(for speaker: Speaker) -> some View {
        Button("Rename…") { renaming = speaker }
        if let first = workspace.segments.first(where: { $0.speakerId == speaker.id }) {
            Button("Play First Segment") { player.play(from: first.start, to: first.end) }
        }
        Button("Show Only This Speaker") { select(speaker.id) }
        let others = workspace.speakers.filter { $0.id != speaker.id }
        if !others.isEmpty {
            Menu("Merge Into") {
                ForEach(others) { other in
                    Button(other.displayName) { merging = (speaker, other) }
                }
            }
        }
    }

    @ViewBuilder
    private var noSpeakers: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let reason = diarizationReason {
                Text("Speakers were not identified for this transcript.")
                    .font(.callout)
                Text(reason).font(.caption).foregroundStyle(.secondary)
                Text("Download the speaker identification model in Settings › Models, then transcribe again.")
                    .font(.caption).foregroundStyle(.secondary)
                OpenSettingsButton()
            } else if workspace.transcript != nil {
                Text("No speakers.").font(.callout).foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var diarizationReason: String? {
        guard let diarization = workspace.transcript?.effective?.lookup("diarization") else {
            return workspace.transcript == nil ? nil : "This transcript was made before speaker identification was available."
        }
        if diarization.lookup("enabled") == .bool(false) {
            return diarization.lookup("reason")?.stringValue ?? "Speaker identification was off."
        }
        return nil
    }

    private var footer: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let diarization = workspace.transcript?.effective?.lookup("diarization"),
               diarization.lookup("enabled") == .bool(true) {
                let found = Int(diarization.lookup("speakers_found")?.doubleValue ?? 0)
                let estimated = Int(diarization.lookup("estimated_speakers")?.doubleValue ?? 0)
                let overlap = diarization.lookup("overlap_seconds")?.doubleValue ?? 0
                Text("\(found) speaker\(found == 1 ? "" : "s") identified automatically"
                     + (estimated > 0 && estimated != found ? " (estimate: \(estimated))" : "")
                     + (overlap >= 1 ? "; \(Timecode.duration(overlap)) of overlapping speech." : "."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if workspace.reviewCount > 0 {
                Button {
                    workspace.filter = .review
                } label: {
                    Label("\(workspace.reviewCount) passage\(workspace.reviewCount == 1 ? "" : "s") to review",
                          systemImage: "questionmark.circle")
                        .font(.caption)
                }
                .buttonStyle(.link)
                .help("Show segments with uncertain speakers, overlapping speech, rejected alignment or your marks")
            }
            Text("Speakers are numbered in order of appearance. Names are only ever the ones you enter.")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct SpeakerRow: View {
    let speaker: Speaker
    let isSpeaking: Bool

    var body: some View {
        HStack(spacing: 8) {
            SpeakerDot(speaker: speaker, size: 11)
                .overlay(Circle().stroke(Color.accentColor, lineWidth: isSpeaking ? 2 : 0).frame(width: 16, height: 16))
            VStack(alignment: .leading, spacing: 1) {
                Text(speaker.displayName).font(.body.weight(.medium)).lineLimit(1)
                Text(details).font(.caption).foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
            if isSpeaking {
                Image(systemName: "waveform").foregroundStyle(Color.accentColor).help("Speaking now")
            }
        }
        .padding(.vertical, 2)
    }

    private var details: String {
        var parts: [String] = []
        if speaker.name != nil { parts.append(speaker.label) }
        parts.append("\(speaker.segmentCount) segment\(speaker.segmentCount == 1 ? "" : "s")")
        if speaker.speakingSeconds > 0 { parts.append(Timecode.duration(speaker.speakingSeconds)) }
        if speaker.origin == "user" { parts.append("added by you") }
        return parts.joined(separator: " · ")
    }
}
