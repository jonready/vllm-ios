// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "vllm-ios",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "VLLMiOS", targets: ["VLLMiOS"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", branch: "main"),
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.6"),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.8.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "VLLMiOS",
            dependencies: [
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "vllm-bench",
            dependencies: [
                "VLLMiOS",
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
    ]
)
