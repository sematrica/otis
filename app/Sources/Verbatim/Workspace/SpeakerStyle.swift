import SwiftUI
import VerbatimCore

/// Stable, distinguishable colours for speakers (by their colour index).
enum SpeakerStyle {
    private static let palette: [Color] = [
        Color(red: 0.20, green: 0.47, blue: 0.85),  // blue
        Color(red: 0.86, green: 0.40, blue: 0.13),  // orange
        Color(red: 0.18, green: 0.62, blue: 0.35),  // green
        Color(red: 0.62, green: 0.32, blue: 0.78),  // purple
        Color(red: 0.80, green: 0.22, blue: 0.36),  // crimson
        Color(red: 0.10, green: 0.60, blue: 0.66),  // teal
        Color(red: 0.62, green: 0.50, blue: 0.10),  // ochre
        Color(red: 0.45, green: 0.45, blue: 0.50),  // grey
    ]

    static func color(_ index: Int?) -> Color {
        guard let index else { return .secondary }
        return palette[((index % palette.count) + palette.count) % palette.count]
    }

    static func color(for speaker: Speaker?) -> Color { color(speaker?.colorIndex) }
}

struct SpeakerDot: View {
    let speaker: Speaker?
    var size: CGFloat = 9

    var body: some View {
        Circle()
            .fill(SpeakerStyle.color(for: speaker))
            .frame(width: size, height: size)
            .accessibilityHidden(true)
    }
}
