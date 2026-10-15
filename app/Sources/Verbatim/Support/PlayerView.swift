import AVKit
import SwiftUI

/// AVKit's AVPlayerView with inline controls.
///
/// Used instead of SwiftUI's `VideoPlayer`: in an app built with SwiftPM,
/// `VideoPlayer` crashes at runtime ("failed to demangle superclass of
/// VideoPlayerView") unless AVKit is linked, and this view references
/// AVPlayerView directly.
struct PlayerView: NSViewRepresentable {
    let player: AVPlayer

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.player = player
        view.controlsStyle = .inline
        view.showsFullScreenToggleButton = true
        view.videoGravity = .resizeAspect
        return view
    }

    func updateNSView(_ view: AVPlayerView, context: Context) {
        if view.player !== player { view.player = player }
    }
}
