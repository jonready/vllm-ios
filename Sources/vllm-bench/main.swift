import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers
import VLLMiOS

// vllm-bench: exercise the VLLMEngine with burst and staggered-arrival
// scenarios on a Mac.
//
//   swift run -c release vllm-bench [--model <hf-id>] [--n 16] [--batch 8]
//       [--gen 128] [--stagger 0.5] [--scenario burst|stagger|both]

func flag(_ name: String, default def: String) -> String {
    let args = CommandLine.arguments
    if let i = args.firstIndex(of: "--\(name)"), i + 1 < args.count { return args[i + 1] }
    return def
}

let modelId = flag("model", default: "mlx-community/Qwen3.5-0.8B-4bit")
let nRequests = Int(flag("n", default: "16")) ?? 16
let maxBatch = Int(flag("batch", default: "8")) ?? 8
let genTokens = Int(flag("gen", default: "128")) ?? 128
let stagger = Double(flag("stagger", default: "0.5")) ?? 0.5
let scenario = flag("scenario", default: "both")

// --dump-prompts <path>: write the workload (system + user prompts) as JSON
// so other engines can benchmark identical inputs.
if let i = CommandLine.arguments.firstIndex(of: "--dump-prompts"), i + 1 < CommandLine.arguments.count {
    let path = CommandLine.arguments[i + 1]
    let payload: [String: Any] = [
        "system": Workloads.systemPrompt,
        "users": Workloads.buildUserPrompts(preset: .medium, count: nRequests),
    ]
    let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted])
    try data.write(to: URL(fileURLWithPath: path))
    print("wrote \(nRequests) prompts to \(path)")
    exit(0)
}

print("vllm-swift bench — model \(modelId), n=\(nRequests), maxBatch=\(maxBatch), gen=\(genTokens)")

let context = try await LLMModelFactory.shared.load(
    from: #hubDownloader(),
    using: #huggingFaceTokenizerLoader(),
    configuration: .init(id: modelId)
) { progress in
    if Int(progress.fractionCompleted * 100) % 25 == 0 {
        print("  download \(Int(progress.fractionCompleted * 100))%")
    }
}
print("model loaded")

// Tokenize the blog-research workload through the model's chat template.
let users = Workloads.buildUserPrompts(preset: .medium, count: nRequests)
var tokenLists: [[Int32]] = []
for user in users {
    let input = UserInput(chat: [.system(Workloads.systemPrompt), .user(user)])
    let lmInput = try await context.processor.prepare(input: input)
    tokenLists.append(lmInput.text.tokens.asArray(Int32.self))
}
let padToken = context.tokenizer.encode(text: "\n").last.map(Int32.init) ?? 198
print("prompts tokenized: \(tokenLists.map(\.count).min()!)–\(tokenLists.map(\.count).max()!) tokens")

let engine = VLLMEngine(model: context.model, padToken: padToken, maxBatch: maxBatch)
engine.decodeChunk = Int(flag("chunk", default: "8")) ?? 8

// --prefix: cache the longest common token prefix (the shared system prompt)
// once, and prefill only suffixes.
var prefixCache: PrefixCache? = nil
if CommandLine.arguments.contains("--prefix") {
    let pfxTokens = PrefixCache.longestCommonPrefix(of: tokenLists)
    if pfxTokens.isEmpty {
        print("prefix: no common prefix found")
    } else {
        let t0p = CFAbsoluteTimeGetCurrent()
        prefixCache = try engine.cachePrefix(pfxTokens)
        print(String(format: "prefix cache: %d tokens, built once in %.2fs", pfxTokens.count, CFAbsoluteTimeGetCurrent() - t0p))
    }
}
engine.log = { print("  [engine] \($0)") }

// Warmup: compile kernels for the shapes in play.
print("warmup…")
let warmInput = try await context.processor.prepare(
    input: UserInput(chat: [.system("You are helpful."), .user("Say OK.")]))
let warmTokens = warmInput.text.tokens.asArray(Int32.self)
let now0 = CFAbsoluteTimeGetCurrent()
_ = try engine.run(requests: (0..<2).map {
    EngineRequest(id: -1 - $0, promptTokens: warmTokens, maxTokens: 4, arrivalTime: now0)
})
print("warmup done\n")

func runScenario(_ name: String, arrivals: [Double]) throws {
    print("=== scenario: \(name) ===")
    let base = CFAbsoluteTimeGetCurrent() + 0.05
    let requests = (0..<nRequests).map { i in
        EngineRequest(
            id: i, promptTokens: tokenLists[i], maxTokens: genTokens,
            arrivalTime: base + arrivals[i])
    }
    let report = try engine.run(requests: requests, prefix: prefixCache)

    print("req    arrive     ttft      e2e   dec t/s  pad waste")
    for r in report.results {
        print(String(
            format: "#%-3d %7.2fs %7.2fs %7.2fs %8.1f %9d",
            r.id, r.arrivalTime - base, r.ttft,
            r.completionTime - r.arrivalTime, r.decodeTPS,
            r.paddedLength - r.promptLength))
    }

    let totalGen = report.results.reduce(0) { $0 + $1.tokens.count }
    let saved = report.results.reduce(0) { $0 + $1.prefixTokens }
    if saved > 0 { print("prefix reuse: \(saved) prompt tokens served from cache") }
    let ttfts = report.results.map(\.ttft).sorted()
    print(String(format: "wall %.1fs | %d gen tokens | agg gen %.1f t/s | ttft p50 %.2fs max %.2fs",
                 report.wallSeconds, totalGen,
                 Double(totalGen) / report.wallSeconds,
                 ttfts[ttfts.count / 2], ttfts.last ?? 0))

    // Step-time by batch size: the scaling curve, measured in situ.
    var byBatch: [Int: (n: Int, total: Double)] = [:]
    for s in report.stepStats {
        byBatch[s.batchSize, default: (0, 0)] = (
            byBatch[s.batchSize, default: (0, 0)].n + 1,
            byBatch[s.batchSize, default: (0, 0)].total + s.stepSeconds)
    }
    print("decode step-time by batch size:")
    for (b, agg) in byBatch.sorted(by: { $0.key < $1.key }) {
        let mean = agg.total / Double(agg.n)
        print(String(format: "  B=%-2d  %6.1f ms/step  → %6.1f tok/s aggregate  (%d steps)",
                     b, mean * 1000, Double(b) / mean, agg.n))
    }

    // Output sanity: first two previews.
    for r in report.results.prefix(2) {
        let text = context.tokenizer.decode(tokenIds: r.tokens.map(Int.init))
        print("  #\(r.id): \(String(text.prefix(110)).replacingOccurrences(of: "\n", with: " "))")
    }
    print()
}

if scenario == "burst" || scenario == "both" {
    try runScenario("burst (all at t=0)", arrivals: Array(repeating: 0, count: nRequests))
}
if scenario == "stagger" || scenario == "both" {
    try runScenario(
        "stagger (one every \(stagger)s)",
        arrivals: (0..<nRequests).map { Double($0) * stagger })
}
