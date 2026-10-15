import Foundation

public enum Timecode {
    /// `HH:MM:SS` (seconds truncated), matching the engine's TXT export.
    public static func clock(_ seconds: Double) -> String {
        let whole = seconds.isFinite && seconds > 0 ? Int(seconds.rounded(.down)) : 0
        return String(format: "%02d:%02d:%02d", whole / 3600, (whole % 3600) / 60, whole % 60)
    }

    /// Short duration for UI: "42 s", "3 min 05 s", "1 h 02 min".
    public static func duration(_ seconds: Double?) -> String {
        guard let seconds, seconds.isFinite, seconds >= 0 else { return "—" }
        let whole = Int(seconds.rounded())
        if whole < 60 { return "\(whole) s" }
        if whole < 3600 { return String(format: "%d min %02d s", whole / 60, whole % 60) }
        return String(format: "%d h %02d min", whole / 3600, (whole % 3600) / 60)
    }
}

public enum ByteSize {
    public static func string(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}

public extension String {
    /// "beamSize" or "beam_size" → "Beam size" (keys in loosely-typed engine records).
    var humanizedKey: String {
        var words: [String] = []
        var current = ""
        for ch in self {
            if ch == "_" || ch == " " {
                if !current.isEmpty { words.append(current) }
                current = ""
                continue
            }
            if ch.isUppercase, !current.isEmpty {
                words.append(current)
                current = String(ch).lowercased()
            } else {
                current.append(ch)
            }
        }
        if !current.isEmpty { words.append(current) }
        guard let first = words.first else { return self }
        return ([first.prefix(1).uppercased() + first.dropFirst()] + words.dropFirst()).joined(separator: " ")
    }
}

/// Locates transcript segments by playback time.
public struct TranscriptIndex: Sendable {
    private let starts: [Double]
    private let ends: [Double]

    public init(segments: [Segment]) {
        starts = segments.map(\.start)
        ends = segments.map(\.end)
    }

    /// Index of the segment playing at `time`: the last segment starting at or
    /// before `time`, provided it has not ended (with a small grace period).
    public func segmentIndex(at time: Double, grace: Double = 0.25) -> Int? {
        var low = 0
        var high = starts.count
        while low < high {
            let mid = (low + high) / 2
            if starts[mid] <= time { low = mid + 1 } else { high = mid }
        }
        let candidate = low - 1
        guard candidate >= 0, time <= ends[candidate] + grace else { return nil }
        return candidate
    }
}
