import Foundation

/// Errors surfaced by the engine client.
public enum EngineClientError: Error, Sendable, Equatable, LocalizedError {
    /// The engine returned a structured error (see docs/ENGINE_PROTOCOL.md).
    case engine(code: String, message: String, data: JSONValue?)
    case notRunning
    case launchFailed(String)
    case terminated(status: Int32)
    case timedOut(method: String)
    case invalidResponse(String)
    case protocolMismatch(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .engine(_, let message, _): return message
        case .notRunning: return "The transcription engine is not running."
        case .launchFailed(let reason): return "The transcription engine could not be started: \(reason)"
        case .terminated(let status): return "The transcription engine stopped unexpectedly (status \(status))."
        case .timedOut(let method): return "The transcription engine did not respond to \(method)."
        case .invalidResponse(let detail): return "Unexpected response from the transcription engine: \(detail)"
        case .protocolMismatch(let expected, let actual):
            return "Engine protocol version \(actual) is not supported (expected \(expected))."
        }
    }

    /// The engine's error code, if this is a structured engine error.
    public var engineCode: String? {
        if case .engine(let code, _, _) = self { return code }
        return nil
    }
}

/// A message pushed by the engine without a request.
public enum EngineNotification: Sendable {
    case jobUpdated(Job)
    case projectUpdated(Project)
    case projectDeleted(String)
    case modelDownload(ModelDownload)
    case other(method: String)
    /// The engine process exited; the client must be started again.
    case engineTerminated(status: Int32)
}

/// How to launch the engine.
public struct EngineConfiguration: Sendable {
    public var executable: URL
    public var arguments: [String]
    public var environment: [String: String]
    /// stderr of the engine is appended here (it must never fill a pipe).
    public var logFile: URL?

    public init(executable: URL, arguments: [String], environment: [String: String], logFile: URL?) {
        self.executable = executable
        self.arguments = arguments
        self.environment = environment
        self.logFile = logFile
    }
}

public let supportedProtocolVersion = 1

/// JSON-lines RPC client for `verbatim-engine serve`.
public actor EngineClient {
    public nonisolated let notifications: AsyncStream<EngineNotification>
    private let notificationSink: AsyncStream<EngineNotification>.Continuation

    private var process: Process?
    private var input: FileHandle?
    private var readerTask: Task<Void, Never>?
    private var pending: [Int: CheckedContinuation<Data, Error>] = [:]
    private var nextID = 1
    private var isStopping = false

    private let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        return encoder
    }()

    public static func makeDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }

    private let decoder = EngineClient.makeDecoder()

    public init() {
        var sink: AsyncStream<EngineNotification>.Continuation!
        notifications = AsyncStream(bufferingPolicy: .unbounded) { sink = $0 }
        notificationSink = sink
    }

    public var isRunning: Bool { process?.isRunning ?? false }

    // MARK: Lifecycle

    /// Launch the engine and verify the protocol version.
    @discardableResult
    public func start(_ configuration: EngineConfiguration, startupTimeout: Duration = .seconds(90)) async throws -> EngineInfo {
        if isRunning { return try await call("engine.info", as: EngineInfo.self) }
        signal(SIGPIPE, SIG_IGN)  // writing to a dead engine must fail, not kill the app

        let process = Process()
        process.executableURL = configuration.executable
        process.arguments = configuration.arguments  // argument vector: no shell involved
        process.environment = configuration.environment
        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        if let logFile = configuration.logFile {
            FileManager.default.createFile(atPath: logFile.path, contents: nil)
            let handle = try FileHandle(forWritingTo: logFile)
            _ = try handle.seekToEnd()
            process.standardError = handle
        } else {
            process.standardError = FileHandle.nullDevice
        }

        let lines = LineStream()
        stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
                lines.finish()
            } else {
                lines.append(data)
            }
        }
        process.terminationHandler = { [weak self] finished in
            let status = finished.terminationStatus
            Task { await self?.handleTermination(status: status) }
        }

        do {
            try process.run()
        } catch {
            throw EngineClientError.launchFailed(error.localizedDescription)
        }
        self.process = process
        self.input = stdinPipe.fileHandleForWriting
        isStopping = false
        readerTask = Task { [weak self] in
            for await line in lines.stream {
                await self?.handle(line: line)
            }
        }

        let info = try await call("engine.info", as: EngineInfo.self, timeout: startupTimeout)
        guard info.protocolVersion == supportedProtocolVersion else {
            await stop()
            throw EngineClientError.protocolMismatch(expected: supportedProtocolVersion, actual: info.protocolVersion)
        }
        return info
    }

    /// Ask the engine to shut down (running jobs become "interrupted" and can
    /// be resumed), then make sure the process is gone.
    public func stop(timeout: Duration = .seconds(15)) async {
        guard let process else { return }
        isStopping = true
        if process.isRunning {
            _ = try? await call("engine.shutdown", as: JSONValue.self, timeout: .seconds(5))
            try? input?.close()
            let deadline = ContinuousClock.now + timeout
            while process.isRunning && ContinuousClock.now < deadline {
                try? await Task.sleep(for: .milliseconds(50))
            }
            if process.isRunning { process.terminate() }
        }
        cleanUp(status: process.isRunning ? -1 : process.terminationStatus, notify: false)
    }

    // MARK: Calls

    public func call<Result: Decodable & Sendable>(
        _ method: String, as type: Result.Type, timeout: Duration? = nil
    ) async throws -> Result {
        try await call(method, params: EmptyParams(), as: type, timeout: timeout)
    }

    public func call<Params: Encodable & Sendable, Result: Decodable & Sendable>(
        _ method: String, params: Params, as type: Result.Type, timeout: Duration? = nil
    ) async throws -> Result {
        guard let input, isRunning else { throw EngineClientError.notRunning }
        let id = nextID
        nextID += 1
        let request = Request(id: id, method: method, params: params)
        var encoded = try encoder.encode(request)
        encoded.append(0x0A)
        let payload = encoded

        let data: Data = try await withCheckedThrowingContinuation { continuation in
            pending[id] = continuation
            do {
                try input.write(contentsOf: payload)
            } catch {
                pending[id] = nil
                continuation.resume(throwing: EngineClientError.notRunning)
                return
            }
            if let timeout {
                Task { [weak self] in
                    try? await Task.sleep(for: timeout)
                    await self?.expire(id: id, method: method)
                }
            }
        }

        let envelope: ResponseEnvelope<Result>
        do {
            envelope = try decoder.decode(ResponseEnvelope<Result>.self, from: data)
        } catch {
            throw EngineClientError.invalidResponse("\(method): \(error)")
        }
        if let failure = envelope.error {
            throw EngineClientError.engine(code: failure.code, message: failure.message, data: failure.data)
        }
        guard let result = envelope.result else {
            throw EngineClientError.invalidResponse("\(method): missing result")
        }
        return result
    }

    // MARK: Internals

    private func expire(id: Int, method: String) {
        guard let continuation = pending.removeValue(forKey: id) else { return }
        continuation.resume(throwing: EngineClientError.timedOut(method: method))
    }

    private func handle(line: Data) {
        guard let head = try? decoder.decode(MessageHead.self, from: line) else { return }
        if let id = head.id {
            pending.removeValue(forKey: id)?.resume(returning: line)
            return
        }
        guard let method = head.method else { return }
        notificationSink.yield(decodeNotification(method: method, line: line))
    }

    private func decodeNotification(method: String, line: Data) -> EngineNotification {
        do {
            switch method {
            case "job.updated":
                return .jobUpdated(try decoder.decode(NotificationEnvelope<Job>.self, from: line).params)
            case "project.updated":
                return .projectUpdated(try decoder.decode(NotificationEnvelope<Project>.self, from: line).params)
            case "project.deleted":
                return .projectDeleted(try decoder.decode(NotificationEnvelope<ProjectDeleted>.self, from: line).params.projectId)
            case "model.download":
                return .modelDownload(try decoder.decode(NotificationEnvelope<ModelDownload>.self, from: line).params)
            default:
                return .other(method: method)
            }
        } catch {
            return .other(method: method)
        }
    }

    private func handleTermination(status: Int32) {
        // An exit we asked for (stop()) is not reported as a failure.
        cleanUp(status: status, notify: !isStopping)
    }

    private func cleanUp(status: Int32, notify: Bool) {
        guard process != nil else { return }
        process = nil
        try? input?.close()
        input = nil
        let waiting = pending
        pending.removeAll()
        for continuation in waiting.values {
            continuation.resume(throwing: EngineClientError.terminated(status: status))
        }
        if notify { notificationSink.yield(.engineTerminated(status: status)) }
    }
}

// MARK: - Wire types

struct EmptyParams: Encodable, Sendable {}

private struct Request<Params: Encodable>: Encodable {
    let id: Int
    let method: String
    let params: Params
}

private struct MessageHead: Decodable {
    let id: Int?
    let method: String?
}

struct ErrorPayload: Decodable, Sendable {
    let code: String
    let message: String
    let data: JSONValue?
}

private struct ResponseEnvelope<Result: Decodable>: Decodable {
    let result: Result?
    let error: ErrorPayload?
}

private struct NotificationEnvelope<Params: Decodable>: Decodable {
    let params: Params
}

/// Splits a byte stream into newline-terminated lines, preserving order.
final class LineStream: @unchecked Sendable {
    let stream: AsyncStream<Data>
    private let continuation: AsyncStream<Data>.Continuation
    private var buffer = Data()
    private let lock = NSLock()

    init() {
        var continuation: AsyncStream<Data>.Continuation!
        stream = AsyncStream(bufferingPolicy: .unbounded) { continuation = $0 }
        self.continuation = continuation
    }

    func append(_ data: Data) {
        lock.lock()
        defer { lock.unlock() }
        buffer.append(data)
        while let newline = buffer.firstIndex(of: 0x0A) {
            let line = buffer.subdata(in: buffer.startIndex..<newline)
            buffer.removeSubrange(buffer.startIndex...newline)
            if !line.isEmpty { continuation.yield(line) }
        }
    }

    func finish() {
        lock.lock()
        defer { lock.unlock() }
        if !buffer.isEmpty { continuation.yield(buffer) }
        buffer.removeAll()
        continuation.finish()
    }
}
