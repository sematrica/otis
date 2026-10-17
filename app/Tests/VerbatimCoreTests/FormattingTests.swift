import XCTest
@testable import VerbatimCore

final class FormattingTests: XCTestCase {
    func testClockMatchesEngineExport() {
        XCTAssertEqual(Timecode.clock(0), "00:00:00")
        XCTAssertEqual(Timecode.clock(72.9), "00:01:12")  // truncated, like the engine
        XCTAssertEqual(Timecode.clock(3 * 3600 + 5), "03:00:05")
        XCTAssertEqual(Timecode.clock(-4), "00:00:00")
        XCTAssertEqual(Timecode.clock(.nan), "00:00:00")
    }

    func testDurations() {
        XCTAssertEqual(Timecode.duration(nil), "—")
        XCTAssertEqual(Timecode.duration(42), "42 s")
        XCTAssertEqual(Timecode.duration(185), "3 min 05 s")
        XCTAssertEqual(Timecode.duration(3725), "1 h 02 min")
    }

    func testHumanizedKeys() {
        XCTAssertEqual("beamSize".humanizedKey, "Beam size")
        XCTAssertEqual("beam_size".humanizedKey, "Beam size")
        XCTAssertEqual("backend".humanizedKey, "Backend")
        XCTAssertEqual("model_id".snakeToCamelCase, "modelId")
    }

    func testTranscriptIndex() {
        func seg(_ id: Int, _ start: Double, _ end: Double) -> Segment {
            Segment(id: id, ordinal: id, start: start, end: end, text: "s\(id)", originalText: "s\(id)", language: nil,
                    avgLogprob: nil, noSpeechProb: nil, compressionRatio: nil, chunkIndex: 0, speakerId: nil, flags: [], words: nil)
        }
        let index = TranscriptIndex(segments: [seg(0, 0, 2), seg(1, 2.5, 5), seg(2, 9, 12)])
        XCTAssertEqual(index.segmentIndex(at: 0), 0)
        XCTAssertEqual(index.segmentIndex(at: 2.2), 0)      // within grace after segment 0
        XCTAssertEqual(index.segmentIndex(at: 3), 1)
        XCTAssertNil(index.segmentIndex(at: 7))             // gap between segments
        XCTAssertEqual(index.segmentIndex(at: 11.9), 2)
        XCTAssertNil(index.segmentIndex(at: -1))
        XCTAssertNil(TranscriptIndex(segments: []).segmentIndex(at: 1))
    }

    func testLineStreamSplitsAcrossChunks() async {
        let lines = LineStream()
        lines.append(Data("{\"a\":1}\n{\"b\"".utf8))
        lines.append(Data(":2}\n\n{\"c\":3}".utf8))
        lines.finish()
        var out: [String] = []
        for await line in lines.stream { out.append(String(decoding: line, as: UTF8.self)) }
        XCTAssertEqual(out, ["{\"a\":1}", "{\"b\":2}", "{\"c\":3}"])
    }

    func testJobOptionsEncodeAsSnakeCase() throws {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = .sortedKeys
        let data = try encoder.encode(JobOptions(mode: "fast", backend: "mlx", modelPath: "/m", chunkSeconds: 30))
        XCTAssertEqual(String(decoding: data, as: UTF8.self),
                       #"{"backend":"mlx","chunk_seconds":30,"mode":"fast","model_path":"\/m"}"#)
    }

    func testLocatorPrefersEnvironmentOverride() throws {
        let python = URL(fileURLWithPath: "/bin/sh")  // any executable file
        let found = EngineLocator.pythonExecutable(environment: [EngineLocator.environmentKey: python.path],
                                                   defaults: UserDefaults(suiteName: UUID().uuidString)!, bundle: .main)
        XCTAssertEqual(found, python)
        let config = EngineLocator.configuration(python: python, dataDirectory: FileManager.default.temporaryDirectory)
        XCTAssertEqual(Array(config.arguments.prefix(4)), ["-I", "-m", "verbatim_engine", "serve"])
        XCTAssertEqual(config.environment["PYTHONNOUSERSITE"], "1")
        XCTAssertNil(config.environment["PYTHONPATH"])
    }

    func testWhyTheEngineStopped() {
        let log = """
        2026-10-17 09:00:01 serve[42] INFO verbatim_engine: starting
        Traceback (most recent call last):
          File "db.py", line 350, in migrate
        verbatim_engine.storage.db.DatabaseTooNew: The library … newer version
        verbatim-engine: cannot start: The library in /x was last opened by a newer version of Verbatim.
        """
        XCTAssertEqual(EngineClient.exitReason(fromLog: log),
                       "The library in /x was last opened by a newer version of Verbatim.")
        let crash = """
        2026-10-17 09:00:01 serve[42] INFO verbatim_engine: starting
        Traceback (most recent call last):
          File "<frozen runpy>", line 198, in _run_module_as_main
            ^^^^^^^^
        ModuleNotFoundError: No module named 'yt_dlp'

        """
        XCTAssertEqual(EngineClient.exitReason(fromLog: crash), "ModuleNotFoundError: No module named 'yt_dlp'")
        // a failure the engine handled earlier is not why it stopped
        let handled = crash + "2026-10-17 09:00:05 serve[42] INFO verbatim_engine: engine stopped\n"
        XCTAssertNil(EngineClient.exitReason(fromLog: handled))
        XCTAssertNil(EngineClient.exitReason(fromLog: ""))
        XCTAssertEqual(EngineClientError.terminated(status: 3, reason: "Disk full.").localizedDescription,
                       "The transcription engine stopped: Disk full.")
        XCTAssertEqual(EngineClientError.terminated(status: 9).localizedDescription,
                       "The transcription engine stopped unexpectedly (status 9).")
    }

    func testAReplyThatIsNotJSONIsStillMatched() {
        XCTAssertEqual(EngineClient.replyID(in: Data(#"{"id":42,"result":{"score":NaN}}"#.utf8)), 42)
        XCTAssertNil(EngineClient.replyID(in: Data(#"{"method":"job.updated","params":{"v":NaN}}"#.utf8)))
        XCTAssertNil(EngineClient.replyID(in: Data(#"{"id":null,"error":{}}"#.utf8)))
        XCTAssertEqual(EngineClientError.invalidResponse("its reply was not valid JSON").localizedDescription,
                       "Unexpected response from the transcription engine: its reply was not valid JSON")
    }
}
