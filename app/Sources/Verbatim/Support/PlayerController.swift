import AVFoundation
import Foundation

/// Owns the AVPlayer for the project workspace.
@MainActor
final class PlayerController: ObservableObject {
    @Published private(set) var player: AVPlayer?
    @Published private(set) var loadError: String?
    private var currentURL: URL?

    func load(_ url: URL?) {
        guard url != currentURL else { return }
        currentURL = url
        player?.pause()
        guard let url else {
            player = nil
            loadError = "The source file is not available."
            return
        }
        guard FileManager.default.fileExists(atPath: url.path) else {
            player = nil
            loadError = "The source file was moved or deleted: \(url.path)"
            return
        }
        loadError = nil
        player = AVPlayer(url: url)
    }

    /// Seek precisely to a transcript timestamp and start playback.
    func play(from seconds: Double) {
        guard let player else { return }
        let time = CMTime(seconds: max(0, seconds), preferredTimescale: 600)
        player.seek(to: time, toleranceBefore: .zero, toleranceAfter: .zero)
        player.play()
    }

    func stop() {
        player?.pause()
    }
}
