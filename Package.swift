// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "MacBoost",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "MacBoost", targets: ["MacBoost"]),
        .library(name: "macboostc", type: .dynamic, targets: ["MacBoostC"]),
        .executable(name: "macboost", targets: ["MacBoostCLI"]),
        .executable(name: "bench", targets: ["bench"]),
    ],
    targets: [
        .target(name: "MacBoost", swiftSettings: [.swiftLanguageMode(.v5)]),
        .target(name: "MacBoostC", dependencies: ["MacBoost"],
                swiftSettings: [.swiftLanguageMode(.v5)]),
        .executableTarget(name: "MacBoostCLI", dependencies: ["MacBoost"],
                          swiftSettings: [.swiftLanguageMode(.v5)]),
        .executableTarget(name: "bench", dependencies: ["MacBoost"],
                          swiftSettings: [.swiftLanguageMode(.v5)]),
        .testTarget(name: "MacBoostTests", dependencies: ["MacBoost"],
                    swiftSettings: [.swiftLanguageMode(.v5)]),
    ]
)
