// swift-tools-version:6.2
import PackageDescription

let package = Package(
    name: "DeepFilterNet-mlx",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "DeepFilterNetMLX", targets: ["DeepFilterNetMLX"]),
        .executable(name: "deepfilternet-mlx", targets: ["deepfilternet-mlx"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", .upToNextMajor(from: "0.30.6")),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", .upToNextMajor(from: "0.6.0")),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.5.0"),
    ],
    targets: [
        .target(
            name: "DeepFilterNetMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "Sources/DeepFilterNetMLX"
        ),
        .executableTarget(
            name: "deepfilternet-mlx",
            dependencies: [
                "DeepFilterNetMLX",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/deepfilternet-mlx-cli"
        ),
        .testTarget(
            name: "DeepFilterNetMLXTests",
            dependencies: ["DeepFilterNetMLX"],
            path: "Tests/DeepFilterNetMLXTests"
        ),
    ]
)
