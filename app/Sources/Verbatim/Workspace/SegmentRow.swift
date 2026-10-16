import SwiftUI
import VerbatimCore

/// What a transcript row can ask the workspace to do.
struct SegmentActions {
    let seek: (Double) -> Void
    let playSegment: () -> Void
    let select: () -> Void
    let assign: (Speaker) -> Void
    let assignNew: () -> Void
    let split: (Int) -> Void
    let mergeWithNext: (() -> Void)?
    let edit: () -> Void
    let markUncertain: () -> Void
    let clearUncertain: () -> Void
    let showOriginal: () -> Void
}

struct SegmentRow: View {
    let segment: Segment
    let speaker: Speaker?
    /// The diarization's speaker, when the user changed it.
    let originalSpeaker: Speaker?
    let speakers: [Speaker]
    let isActive: Bool
    let isSelected: Bool
    /// Playback position while this segment is playing (for word highlighting).
    let playbackTime: Double?
    let actions: SegmentActions

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            VStack(alignment: .leading, spacing: 3) {
                Button {
                    actions.seek(segment.start)
                } label: {
                    Text(Timecode.clock(segment.start))
                        .font(.callout.monospacedDigit())
                        .foregroundStyle(Color.accentColor)
                }
                .buttonStyle(.plain)
                .help("Play from here")
                speakerMenu
            }
            .frame(width: 118, alignment: .leading)
            VStack(alignment: .leading, spacing: 4) {
                if isSelected, let words = segment.words, !words.isEmpty {
                    WordFlow(words: words, playbackTime: playbackTime, onSeek: actions.seek, onSplit: actions.split)
                } else {
                    Text(attributedText)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                FlagStrip(segment: segment, originalSpeaker: originalSpeaker, speakers: speakers)
            }
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
        .onTapGesture(count: 2) { actions.seek(segment.start) }
        .onTapGesture { actions.select() }
        .contextMenu { contextMenu }
    }

    // MARK: Speaker

    private var speakerMenu: some View {
        Menu {
            ForEach(speakers) { candidate in
                Button {
                    actions.assign(candidate)
                } label: {
                    Text(candidate.id == segment.speakerId ? "✓ \(candidate.displayName)" : candidate.displayName)
                }
                .disabled(candidate.id == segment.speakerId)
            }
            Divider()
            Button("New Speaker…", action: actions.assignNew)
        } label: {
            HStack(spacing: 5) {
                SpeakerDot(speaker: speaker)
                Text(speaker?.displayName ?? "Unknown speaker")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(speaker == nil ? Color.secondary : Color.primary)
                    .lineLimit(1)
            }
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Speaker of this segment — click to change")
    }

    // MARK: Text

    /// The text, with the word being played highlighted.
    private var attributedText: AttributedString {
        guard let playbackTime, let words = segment.words, !words.isEmpty else {
            return AttributedString(segment.text.trimmingCharacters(in: .whitespaces))
        }
        var result = AttributedString()
        for (i, word) in words.enumerated() {
            var piece = AttributedString(i == 0 ? word.text.trimmingCharacters(in: .whitespaces) : word.text)
            if let start = word.start, let end = word.end, playbackTime >= start, playbackTime < max(end, start + 0.05) {
                piece[AttributeScopes.SwiftUIAttributes.BackgroundColorAttribute.self] = Color.accentColor.opacity(0.35)
            }
            result += piece
        }
        return result
    }

    // MARK: Context menu

    @ViewBuilder
    private var contextMenu: some View {
        Button("Play Segment", action: actions.playSegment)
        Button("Play from Here") { actions.seek(segment.start) }
        Divider()
        Button("Edit Text…", action: actions.edit)
        if segment.isTextEdited {
            Button("Show Recognised Text", action: actions.showOriginal)
        }
        if let words = segment.words, words.count > 1 {
            Menu("Split Before Word") {
                ForEach(Array(words.enumerated().dropFirst()), id: \.offset) { item in
                    Button(item.element.text.trimmingCharacters(in: .whitespaces)) { actions.split(item.offset) }
                }
            }
        }
        if let merge = actions.mergeWithNext {
            Button("Merge with Next Segment", action: merge)
        }
        Divider()
        Menu("Speaker") {
            ForEach(speakers) { candidate in
                Button(candidate.displayName) { actions.assign(candidate) }
                    .disabled(candidate.id == segment.speakerId)
            }
            Divider()
            Button("New Speaker…", action: actions.assignNew)
        }
        Divider()
        if segment.flagTypes.contains("marked_uncertain") {
            Button("Clear Uncertain Mark", action: actions.clearUncertain)
        } else {
            Button("Mark as Uncertain…", action: actions.markUncertain)
        }
    }
}

/// Words of the selected segment as buttons: click to play from a word,
/// right-click to split the segment before it.
struct WordFlow: View {
    let words: [Word]
    let playbackTime: Double?
    let onSeek: (Double) -> Void
    let onSplit: (Int) -> Void

    var body: some View {
        FlowLayout(spacing: 4, lineSpacing: 3) {
            ForEach(Array(words.enumerated()), id: \.offset) { item in
                let word = item.element
                Button {
                    if let start = word.start { onSeek(start) }
                } label: {
                    Text(word.text.trimmingCharacters(in: .whitespaces))
                        .underline(word.timingSource == "interpolated", color: .secondary)
                        .padding(.horizontal, 2)
                        .background(isPlaying(word) ? Color.accentColor.opacity(0.35) : Color.clear,
                                    in: RoundedRectangle(cornerRadius: 3))
                }
                .buttonStyle(.plain)
                .help(help(for: word))
                .contextMenu {
                    if let start = word.start {
                        Button("Play from “\(word.text.trimmingCharacters(in: .whitespaces))”") { onSeek(start) }
                    }
                    if item.offset > 0 {
                        Button("Split Segment Before This Word") { onSplit(item.offset) }
                    }
                }
            }
        }
    }

    private func isPlaying(_ word: Word) -> Bool {
        guard let playbackTime, let start = word.start, let end = word.end else { return false }
        return playbackTime >= start && playbackTime < max(end, start + 0.05)
    }

    private func help(for word: Word) -> String {
        guard let start = word.start else { return "No timing" }
        let source: String
        switch word.timingSource {
        case "alignment": source = "aligned to the audio"
        case "wildcard": source = "timed between its neighbours (not spelled out by the aligner)"
        case "interpolated": source = "estimated (edited word)"
        default: source = "Whisper's estimate"
        }
        return String(format: "%.2f s — %@", start, source)
    }
}

/// Icons for everything the engine flagged about a segment.
struct FlagStrip: View {
    let segment: Segment
    let originalSpeaker: Speaker?
    let speakers: [Speaker]

    var body: some View {
        let items = flags
        if !items.isEmpty {
            HStack(spacing: 8) {
                ForEach(items, id: \.label) { item in
                    Label(item.label, systemImage: item.symbol)
                        .labelStyle(.titleAndIcon)
                        .font(.caption2)
                        .foregroundStyle(item.color)
                        .help(item.detail)
                }
            }
        }
    }

    private struct Item {
        let symbol: String
        let label: String
        let detail: String
        let color: Color
    }

    private func names(_ ids: [JSONValue]?) -> String {
        let names = (ids ?? []).compactMap { value -> String? in
            guard let id = value.doubleValue.map({ Int($0) }) else { return nil }
            return speakers.first { $0.id == id }?.displayName
        }
        return names.isEmpty ? "another speaker" : names.joined(separator: ", ")
    }

    private var flags: [Item] {
        var out: [Item] = []
        if let flag = segment.flag("marked_uncertain") {
            let note = flag.lookup("note")?.stringValue
            out.append(Item(symbol: "flag.fill", label: "Marked uncertain",
                            detail: note.map { "Marked by you: \($0)" } ?? "Marked by you as uncertain.", color: .orange))
        }
        if let flag = segment.flag("overlapping_speech") {
            var others: [JSONValue]? = nil
            if case .some(.array(let values)) = flag.lookup("other_speakers") { others = values }
            out.append(Item(symbol: "person.2.wave.2", label: "Overlap",
                            detail: "Several people talk at once here (with \(names(others))). Words in overlaps are often incomplete.",
                            color: .purple))
        }
        if let flag = segment.flag("speaker_uncertain") {
            var reasons: [String] = []
            if case .some(.array(let values)) = flag.lookup("reasons") { reasons = values.compactMap(\.stringValue) }
            let text = reasons.map { reason -> String in
                switch reason {
                case "low_coverage": return "the speaker is detected in only part of this speech"
                case "windows_disagree": return "the analysis is inconsistent here"
                case "similar_to_other_speaker": return "the voice is close to another speaker's"
                case "no_speech_under_words": return "no voice was detected under these words; the nearest speaker was used"
                default: return reason
                }
            }.joined(separator: "; ")
            out.append(Item(symbol: "questionmark.circle", label: "Speaker uncertain",
                            detail: "The speaker of this segment may be wrong: \(text).", color: .orange))
        } else if segment.flag("speaker_change_uncertain") != nil {
            out.append(Item(symbol: "questionmark.circle", label: "Speaker change",
                            detail: "The change of speaker may be off by a word here.", color: .orange))
        }
        if segment.flag("speaker_unassigned") != nil {
            out.append(Item(symbol: "person.crop.circle.badge.questionmark", label: "No speaker",
                            detail: "No voice was detected near these words, so no speaker could be assigned.", color: .orange))
        }
        if let flag = segment.flag("alignment_failed") {
            let reason = flag.lookup("reason")?.stringValue ?? "alignment failed"
            out.append(Item(symbol: "clock.badge.exclamationmark", label: "Approximate word times",
                            detail: "Word alignment was rejected (\(reason)); Whisper's word times are shown.", color: .secondary))
        }
        if segment.isTextEdited {
            out.append(Item(symbol: "pencil", label: "Edited",
                            detail: "You edited this text. Recognised: “\(segment.originalText.trimmingCharacters(in: .whitespaces))”",
                            color: .blue))
        }
        if let originalSpeaker {
            out.append(Item(symbol: "person.crop.circle.badge.checkmark", label: "Speaker corrected",
                            detail: "Originally identified as \(originalSpeaker.displayName).", color: .blue))
        }
        if segment.flag("boundary_duplicate_removed") != nil {
            let text = segment.flag("boundary_duplicate_removed")?.lookup("text")?.stringValue ?? ""
            out.append(Item(symbol: "scissors", label: "Joined",
                            detail: "A repeated passage at a processing-chunk boundary was removed: “\(text)”.", color: .secondary))
        }
        return out
    }
}

/// Lays out children left to right, wrapping to new lines (macOS 13 Layout).
struct FlowLayout: Layout {
    var spacing: CGFloat = 4
    var lineSpacing: CGFloat = 2

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let maxWidth = proposal.width ?? .greatestFiniteMagnitude
        var x: CGFloat = 0
        var y: CGFloat = 0
        var lineHeight: CGFloat = 0
        var widest: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > 0, x + size.width > maxWidth {
                y += lineHeight + lineSpacing
                x = 0
                lineHeight = 0
            }
            x += size.width + spacing
            lineHeight = max(lineHeight, size.height)
            widest = max(widest, x - spacing)
        }
        return CGSize(width: proposal.width ?? widest, height: y + lineHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX
        var y = bounds.minY
        var lineHeight: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > bounds.minX, x + size.width > bounds.maxX {
                y += lineHeight + lineSpacing
                x = bounds.minX
                lineHeight = 0
            }
            subview.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
            x += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }
    }
}
