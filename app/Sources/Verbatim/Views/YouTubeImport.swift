import SwiftUI
import VerbatimCore

/// Checks a YouTube link with the engine and, once the user confirms they are
/// authorized to retrieve the video, creates the project and queues its
/// retrieval and transcription with the import panel's settings.
struct YouTubeImportSheet: View {
    let link: String
    let options: JobOptions
    let onCreated: (String) -> Void
    @EnvironmentObject private var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var inspection: YouTubeInspection?
    @State private var error: String?
    @State private var authorized = false
    @State private var audioOnly = false
    @State private var captions: Set<String> = []
    @State private var isCreating = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Import from YouTube").font(.title3.weight(.semibold))
            if let error {
                Label(error, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            } else if let inspection {
                details(inspection)
            } else {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Asking YouTube about this video…").foregroundStyle(.secondary)
                }
            }
            Spacer(minLength: 0)
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Retrieve and Transcribe") { Task { await create() } }
                    .keyboardShortcut(.defaultAction)
                    .disabled(inspection == nil || inspection?.refusal != nil || !authorized || isCreating)
            }
        }
        .padding(20)
        .frame(width: 560, height: 520)
        .task { await inspect() }
    }

    @ViewBuilder
    private func details(_ video: YouTubeInspection) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(video.title ?? "Untitled video").font(.headline).lineLimit(2)
            Text([video.channel, video.duration.map { Timecode.duration($0) }].compactMap { $0 }.joined(separator: " · "))
                .foregroundStyle(.secondary)
            Text(video.license.map { "Licence: \($0)" } ?? "Licence: not stated by YouTube")
                .font(.caption).foregroundStyle(.secondary)
        }
        if let refusal = video.refusal {
            Label(refusal.message, systemImage: "hand.raised").foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
        } else {
            Picker("Retrieve", selection: $audioOnly) {
                Text("Video (up to 720p) and audio" + size(video.approxBytes?.video)).tag(false)
                Text("Audio only" + size(video.approxBytes?.audio)).tag(true)
            }
            .pickerStyle(.radioGroup)
            if video.captions.isEmpty {
                Text("No captions are published with this video.").font(.callout).foregroundStyle(.secondary)
            } else {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Captions to keep alongside the transcript (compared with it, never mixed into it):")
                        .font(.callout)
                    ForEach(video.captions) { track in
                        Toggle(isOn: Binding(
                            get: { captions.contains(track.id) },
                            set: { if $0 { captions.insert(track.id) } else { captions.remove(track.id) } }
                        )) {
                            Text("\(track.name ?? track.language)"
                                 + (track.isAutomatic ? " — automatic (YouTube's speech recognition)" : " — by the creator"))
                        }
                        .toggleStyle(.checkbox)
                    }
                }
            }
            if !video.sponsorChapters.isEmpty {
                Text("The creator marked \(video.sponsorChapters.count) sponsor chapter(s); passages in them will be labelled.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Text(video.adsNote).font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Toggle(isOn: $authorized) {
                Text("I am authorized to download and transcribe this video: it is my own, it is licensed for "
                     + "reuse, or I have the rights holder's permission.")
                    .fixedSize(horizontal: false, vertical: true)
            }
            .toggleStyle(.checkbox)
            Text("Verbatim connects to YouTube only to retrieve this video. It does not sign in, and does not "
                 + "bypass DRM or restrictions.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private func size(_ bytes: Int?) -> String {
        bytes.map { " — about \(ByteSize.string(Int64($0)))" } ?? ""
    }

    private func inspect() async {
        do {
            inspection = try await model.client.inspectYouTube(link)
            // creator captions are chosen by default; automatic ones on request
            captions = Set(inspection?.captions.filter { !$0.isAutomatic }.map(\.id) ?? [])
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func create() async {
        guard let video = inspection else { return }
        isCreating = true
        defer { isCreating = false }
        let chosen = video.captions.filter { captions.contains($0.id) }
            .map { CaptionChoice(language: $0.language, kind: $0.kind) }
        do {
            let created = try await model.client.createFromYouTube(video.url, authorized: authorized, options: options,
                                                                   audioOnly: audioOnly, captions: chosen)
            model.noteCreated(created)
            dismiss()
            onCreated(created.project.id)
        } catch {
            self.error = error.localizedDescription
        }
    }
}
