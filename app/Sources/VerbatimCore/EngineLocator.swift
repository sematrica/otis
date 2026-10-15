import Foundation

/// Finds the Python runtime that hosts the engine.
///
/// Order: `VERBATIM_ENGINE_PYTHON` environment variable (development and
/// tests), a path chosen in Settings, then the runtime embedded in the app
/// bundle at `Contents/Resources/python/bin/python3`.
public enum EngineLocator {
    public static let environmentKey = "VERBATIM_ENGINE_PYTHON"
    public static let defaultsKey = "EnginePythonPath"

    public static func pythonExecutable(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        defaults: UserDefaults = .standard,
        bundle: Bundle = .main
    ) -> URL? {
        let fm = FileManager.default
        if let path = environment[environmentKey], fm.isExecutableFile(atPath: path) {
            return URL(fileURLWithPath: path)
        }
        if let path = defaults.string(forKey: defaultsKey), !path.isEmpty, fm.isExecutableFile(atPath: path) {
            return URL(fileURLWithPath: path)
        }
        if let resources = bundle.resourceURL {
            let embedded = resources.appendingPathComponent("python/bin/python3")
            if fm.isExecutableFile(atPath: embedded.path) { return embedded }
        }
        return nil
    }

    /// Default application data directory (shared with the engine).
    public static func dataDirectory() -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support")
        return base.appendingPathComponent("Verbatim", isDirectory: true)
    }

    public static func configuration(
        python: URL,
        dataDirectory: URL? = nil,
        extraEnvironment: [String: String] = [:]
    ) -> EngineConfiguration {
        let dataDir = dataDirectory ?? Self.dataDirectory()
        try? FileManager.default.createDirectory(at: dataDir.appendingPathComponent("logs"), withIntermediateDirectories: true)
        let inherited = ProcessInfo.processInfo.environment
        var environment: [String: String] = [:]
        // A minimal environment: never leak arbitrary variables (e.g. a user's
        // PYTHONPATH) into the embedded runtime.
        for key in ["HOME", "USER", "LOGNAME", "TMPDIR", "LANG", "LC_ALL", "PATH"] {
            if let value = inherited[key] { environment[key] = value }
        }
        environment["PATH"] = environment["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin"
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONNOUSERSITE"] = "1"
        environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
        for (key, value) in extraEnvironment { environment[key] = value }
        return EngineConfiguration(
            executable: python,
            arguments: ["-I", "-m", "verbatim_engine", "serve", "--data-dir", dataDir.path],
            environment: environment,
            logFile: dataDir.appendingPathComponent("logs/engine-stderr.log")
        )
    }
}
