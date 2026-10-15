// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "Verbatim",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "Verbatim", targets: ["Verbatim"]),
        .library(name: "VerbatimCore", targets: ["VerbatimCore"]),
    ],
    targets: [
        // Foundation-only: engine client, protocol types, formatting.
        .target(name: "VerbatimCore"),
        // SwiftUI / AVKit application.
        .executableTarget(
            name: "Verbatim",
            dependencies: ["VerbatimCore"],
            linkerSettings: [.linkedFramework("AVKit"), .linkedFramework("AVFoundation")]
        ),
        .testTarget(name: "VerbatimCoreTests", dependencies: ["VerbatimCore"]),
    ]
)
