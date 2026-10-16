import AppKit
import SwiftUI

/// The single application model, shared by the window, the Settings scene and
/// the app delegate (which must stop the engine on quit).
@MainActor
enum AppEnvironment {
    static let model = AppModel()
}

@main
struct VerbatimApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup("Verbatim") {
            ContentView()
                .environmentObject(AppEnvironment.model)
                .frame(minWidth: 980, minHeight: 620)
        }
        .commands {
            CommandGroup(replacing: .newItem) {}
            TranscriptMenuCommands()
        }

        Settings {
            SettingsView()
                .environmentObject(AppEnvironment.model)
                .frame(width: 760, height: 620)
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var isTerminating = false
    @MainActor private var model: AppModel { AppEnvironment.model }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Needed when launched as a bare executable (swift run); harmless in a bundle.
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        if let selfTest = SelfTest.fromEnvironment() {
            Task { await selfTest.run(model) }
        } else {
            Task { await model.start() }
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        if isTerminating { return .terminateNow }  // a second quit request: don't wait again
        isTerminating = true
        // Let the engine stop cleanly: a running job becomes "interrupted" and
        // can be resumed on the next launch. While AppKit waits for the reply it
        // runs a nested run loop that cannot drain the main queue if terminate
        // was itself called from main-queue work, so neither the shutdown nor
        // the reply may depend on the main actor.
        let client = model.prepareForTermination()
        Task.detached {
            await client.stop()
            RunLoop.main.perform(inModes: [.common]) {
                MainActor.assumeIsolated { NSApp.reply(toApplicationShouldTerminate: true) }
            }
        }
        return .terminateLater
    }
}
