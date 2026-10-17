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
    /// Re-transcribe this passage and let the audio decide its wording.
    let recheck: (() -> Void)?
    let remove: () -> Void
    let setReviewed: (Bool) -> Void
    /// Use a wording recorded by re-checking ("first_pass" or a candidate's source).
    let chooseWording: (String) -> Void
}

/// Below this recogniser probability a word is shown as uncertain (the same
/// threshold the engine's quality check uses).
let lowConfidenceProbability = 0.5

extension Word {
    var isLowConfidence: Bool { (probability ?? 1) < lowConfidenceProbability }
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

    /// The text, with the word being played highlighted and words the
    /// recogniser was unsure of in orange.
    private var attributedText: AttributedString {
        guard let words = segment.words, !words.isEmpty else {
            return AttributedString(segment.text.trimmingCharacters(in: .whitespaces))
        }
        var result = AttributedString()
        for (i, word) in words.enumerated() {
            var piece = AttributedString(i == 0 ? word.text.trimmingCharacters(in: .whitespaces) : word.text)
            if word.isLowConfidence {
                piece[AttributeScopes.SwiftUIAttributes.ForegroundColorAttribute.self] = Color.orange
            }
            if let playbackTime, let start = word.start, let end = word.end,
               playbackTime >= start, playbackTime < max(end, start + 0.05) {
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
        let wordings = WordingChoice.all(in: segment)
        if !wordings.isEmpty {
            Menu("Use Wording") {
                ForEach(wordings) { choice in
                    Button("\(choice.label): “\(choice.text)”") { actions.chooseWording(choice.source) }
                        .disabled(choice.text == segment.text.trimmingCharacters(in: .whitespaces))
                }
            }
        }
        if let recheck = actions.recheck {
            Button("Re-check Passage", action: recheck)
                .help("Decode this passage again and keep the wording the audio supports best")
        }
        if segment.isReviewed {
            Button("Mark as Needing Review") { actions.setReviewed(false) }
        } else {
            Button("Mark as Reviewed") { actions.setReviewed(true) }
        }
        if segment.flagTypes.contains("marked_uncertain") {
            Button("Clear Uncertain Mark", action: actions.clearUncertain)
        } else {
            Button("Mark as Uncertain…", action: actions.markUncertain)
        }
        Divider()
        Button("Remove Passage…", role: .destructive, action: actions.remove)
    }
}

/// A wording of a passage recorded by re-checking.
struct WordingChoice: Identifiable, Hashable {
    let source: String
    let label: String
    let text: String
    let score: Double?
    var id: String { source }

    static func label(for source: String) -> String {
        switch source {
        case "first_pass": return "First pass"
        case "context": return "Re-decoded with context"
        case "plain": return "Re-decoded"
        case "vad": return "Re-decoded (speech only)"
        default: return source
        }
    }

    /// Distinct wordings from the segment's re-check flags, first pass first.
    static func all(in segment: Segment) -> [WordingChoice] {
        var out: [WordingChoice] = []
        for type in ["retranscribed", "alternatives", "recovered_speech"] {
            guard let flag = segment.flag(type) else { continue }
            if let first = flag.lookup("first_pass")?.stringValue, !first.isEmpty, !out.contains(where: { $0.text == first }) {
                out.append(WordingChoice(source: "first_pass", label: label(for: "first_pass"), text: first, score: nil))
            }
            if case .some(.array(let candidates)) = flag.lookup("candidates") {
                for candidate in candidates {
                    guard let source = candidate.lookup("source")?.stringValue,
                          let text = candidate.lookup("text")?.stringValue, !text.isEmpty,
                          !out.contains(where: { $0.text == text }) else { continue }
                    out.append(WordingChoice(source: source, label: label(for: source), text: text,
                                             score: candidate.lookup("score")?.doubleValue))
                }
            }
        }
        return out
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
                        .underline(word.isLowConfidence || word.timingSource == "interpolated", pattern: .dot,
                                   color: word.isLowConfidence ? .orange : .secondary)
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
        let confidence = word.probability.map { " · recogniser confidence \(Int(($0 * 100).rounded())) %" } ?? ""
        guard let start = word.start else { return "No timing" + confidence }
        let source: String
        switch word.timingSource {
        case "alignment": source = "aligned to the audio"
        case "wildcard": source = "timed between its neighbours (not spelled out by the aligner)"
        case "interpolated": source = "estimated (edited word)"
        default: source = "Whisper's estimate"
        }
        return String(format: "%.2f s — %@", start, source) + confidence
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
        out += qualityItems
        if segment.isReviewed {
            out.append(Item(symbol: "checkmark.circle", label: "Reviewed",
                            detail: "You marked this passage as checked.", color: .green))
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

    private func strings(_ value: JSONValue?) -> [String] {
        if case .some(.array(let values)) = value { return values.compactMap(\.stringValue) }
        return []
    }

    private static let hallucinationReasons = [
        "no_speech_detected": "no speech was detected under these words",
        "text_not_in_audio": "the acoustic model could not find these words in the audio",
        "repetition_loop": "the recogniser repeated itself",
        "high_no_speech_probability": "the recogniser itself judged this likely to be silence",
        "known_phrase": "it is a phrase Whisper often produces on silence or music",
        "no_acoustic_support": "re-checking found the audio better explained by no speech than by these words",
    ]
    private static let unclearReasons = [
        "low_average_probability": "the recogniser's probability is low",
        "many_uncertain_words": "many words are uncertain",
        "decoder_fallback": "the recogniser had to retry with sampling",
    ]

    /// Stage 3: quality checks and re-checking.
    private var qualityItems: [Item] {
        var out: [Item] = []
        if let flag = segment.flag("possible_hallucination") {
            let reasons = strings(flag.lookup("reasons")).map { Self.hallucinationReasons[$0] ?? $0 }
            let high = flag.lookup("severity")?.stringValue == "high"
            var detail = "These words may not have been spoken: \(reasons.joined(separator: "; ")). "
            if strings(flag.lookup("reasons")).contains("no_acoustic_support"),
               (flag.lookup("speech_coverage")?.doubleValue ?? 0) >= 0.5 {
                detail += "Speech was detected here, so the acoustic model may simply not recognise it "
                    + "(overlapping, distant or telephone speech). "
            }
            detail += "Listen before keeping them; Remove Passage deletes them (undoable)."
            out.append(Item(symbol: "exclamationmark.bubble", label: "Possible hallucination",
                            detail: detail, color: high ? .red : .orange))
        }
        if let flag = segment.flag("low_confidence") {
            let reasons = strings(flag.lookup("reasons")).map { Self.unclearReasons[$0] ?? $0 }
            var detail = "Unclear speech: \(reasons.joined(separator: "; "))."
            if flag.lookup("confirmed_acoustically") == .bool(true) {
                detail += " Re-checking compared other wordings and the audio supports this one."
            }
            out.append(Item(symbol: "waveform.badge.exclamationmark", label: "Unclear", detail: detail,
                            color: flag.lookup("severity")?.stringValue == "low" ? .secondary : .orange))
        }
        if let flag = segment.flag("retranscribed") {
            let first = flag.lookup("first_pass")?.stringValue ?? ""
            out.append(Item(symbol: "arrow.triangle.2.circlepath", label: "Re-worded",
                            detail: "Re-checking replaced the first-pass wording “\(first)” because the audio clearly "
                                + "supports the new one. Use Wording › First pass to restore it.", color: .blue))
        }
        if let flag = segment.flag("recovered_speech"), flag.lookup("resolved") != .bool(true) {
            out.append(Item(symbol: "plus.bubble", label: "Recovered",
                            detail: "The first pass had no words here; re-checking found these, supported by the audio.",
                            color: .blue))
        }
        if let flag = segment.flag("alternatives"), flag.lookup("resolved") != .bool(true) {
            let texts = WordingChoice.all(in: segment).filter { $0.source != "first_pass" }.map { "“\($0.text)”" }
            let why = flag.lookup("reason")?.stringValue.map { " (\($0))" } ?? ""
            out.append(Item(symbol: "text.badge.plus", label: "Alternatives",
                            detail: "Other wordings the audio supports about as well\(why): \(texts.joined(separator: ", ")). "
                                + "Choose one with Use Wording, or keep this one.", color: .orange))
        }
        if let flag = segment.flag("missing_speech") {
            let start = flag.lookup("start")?.doubleValue ?? segment.end
            let end = flag.lookup("end")?.doubleValue ?? start
            out.append(Item(symbol: "ear.trianglebadge.exclamationmark", label: "Untranscribed speech",
                            detail: "Speech was detected from \(Timecode.clock(start)) to \(Timecode.clock(end)) but no words "
                                + "were transcribed there.", color: .orange))
        }
        if let flag = segment.flag("repeated_words") {
            var repeats: [String] = []
            if case .some(.array(let values)) = flag.lookup("repeats") {
                repeats = values.compactMap { $0.lookup("text")?.stringValue }
            }
            out.append(Item(symbol: "repeat", label: "Repeated words",
                            detail: "Repeated: \(repeats.map { "“\($0)”" }.joined(separator: ", ")). Often genuine in speech.",
                            color: .secondary))
        }
        if let flag = segment.flag("possible_duplicate") {
            let text = flag.lookup("text")?.stringValue ?? ""
            out.append(Item(symbol: "doc.on.doc", label: "Possible duplicate",
                            detail: "“\(text)” also ends the previous passage; check that it was said twice.", color: .orange))
        }
        if segment.flag("timing_inconsistent") != nil {
            out.append(Item(symbol: "clock.arrow.circlepath", label: "Word order",
                            detail: "Some word times are out of order here.", color: .orange))
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
