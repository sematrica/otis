import SwiftUI
import VerbatimCore

/// The user's choice about speaker identification for a new transcription.
struct SpeakerChoice: Equatable {
    var identify = true
    /// Known number of speakers; 0 = estimate automatically.
    var count = 0

    var tristate: Tristate { identify ? .on : .off }
    var bounds: SpeakerBounds? { identify && count > 0 ? SpeakerBounds(num: count) : nil }
}

struct SpeakerOptionsView: View {
    @Binding var choice: SpeakerChoice

    var body: some View {
        HStack(spacing: 14) {
            Toggle("Identify speakers", isOn: $choice.identify)
                .help("Label who speaks when (speaker diarization). Runs on this Mac.")
            Picker("Number of speakers", selection: $choice.count) {
                Text("Estimate automatically").tag(0)
                ForEach(1...12, id: \.self) { n in Text("\(n)").tag(n) }
            }
            .frame(width: 290)
            .disabled(!choice.identify)
            .help("If you know how many people speak, say so; otherwise it is estimated")
        }
    }
}
