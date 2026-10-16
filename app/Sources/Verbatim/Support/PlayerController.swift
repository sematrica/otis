import AVFoundation
import Foundation

/// Owns the AVPlayer for the project workspace and publishes the playback
/// position so the transcript can follow it.
@MainActor
final class PlayerController: ObservableObject {
    @Published private(set) var player: AVPlayer?
    @Published private(set) var loadError: String?
    /// Playback position in seconds, updated ten times per second.
    @Published private(set) var currentTime: Double = 0
    @Published private(set) var isPlaying = false
    private var currentURL: URL?
    private var timeObserver: Any?
    private weak var observedPlayer: AVPlayer?
    /// Pause when playback reaches this time ("play this segment").
    private var stopTime: Double?

    func load(_ url: URL?) {
        guard url != currentURL else { return }
        currentURL = url
        removeObserver()
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
        let newPlayer = AVPlayer(url: url)
        player = newPlayer
        addObserver(to: newPlayer)
    }

    /// Seek precisely to a transcript timestamp and start playback.
    func play(from seconds: Double) {
        stopTime = nil
        seek(to: seconds, andPlay: true)
    }

    /// Play only [start, end) - "replay the associated audio".
    func play(from start: Double, to end: Double) {
        stopTime = end
        seek(to: start, andPlay: true)
    }

    func seek(to seconds: Double, andPlay play: Bool = false) {
        guard let player else { return }
        let time = CMTime(seconds: max(0, seconds), preferredTimescale: 600)
        player.seek(to: time, toleranceBefore: .zero, toleranceAfter: .zero)
        currentTime = max(0, seconds)
        if play { player.play() }
    }

    func togglePlayback() {
        guard let player else { return }
        if player.rate != 0 {
            player.pause()
        } else {
            stopTime = nil
            player.play()
        }
    }

    func stop() {
        player?.pause()
        removeObserver()
        currentURL = nil
    }

    private func addObserver(to player: AVPlayer) {
        let interval = CMTime(seconds: 0.1, preferredTimescale: 600)
        timeObserver = player.addPeriodicTimeObserver(forInterval: interval, queue: .main) { [weak self] time in
            // Delivered on the main queue.
            MainActor.assumeIsolated {
                self?.tick(time.seconds)
            }
        }
        observedPlayer = player
    }

    private func removeObserver() {
        if let timeObserver, let observedPlayer {
            observedPlayer.removeTimeObserver(timeObserver)
        }
        timeObserver = nil
        observedPlayer = nil
    }

    private func tick(_ seconds: Double) {
        guard seconds.isFinite else { return }
        currentTime = seconds
        let playing = (player?.rate ?? 0) != 0
        if playing != isPlaying { isPlaying = playing }
        if let stopTime, seconds >= stopTime {
            self.stopTime = nil
            player?.pause()
        }
    }
}
