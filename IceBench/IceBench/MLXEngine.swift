import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers


// Single-stream MLX arm. MLX has no batched multi-sequence decode, so this
// benchmarks the c1-equivalent path only: sequential requests, greedy sampling,
// same workload and metrics as the llama.cpp arm.
final class MLXEngine {
    static let shared = MLXEngine()
    private var container: ModelContainer?
    private(set) var modelId = ""

    static let presets: [(label: String, id: String)] = [
        ("4bit (uniform)", "mlx-community/Qwen3.5-0.8B-4bit"),
        ("OptiQ-4bit (dynamic 4/8)", "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"),
        ("mixed_4_6", "mlx-community/Qwen3.5-0.8B-mixed_4_6"),
        ("8bit (quality ceiling)", "mlx-community/Qwen3.5-0.8B-MLX-8bit"),
    ]

    private init() {}

    var isLoaded: Bool { container != nil }

    func load(modelId newId: String, progress: @escaping (String) -> Void) async throws {
        if container != nil && modelId == newId { return }
        container = nil
        modelId = newId
        // iOS: MLX's freed-buffer cache is unbounded by default; on iPhone the
        // jetsam limit (~3 GB) counts cached buffers, so cap it hard.
        MLX.GPU.set(cacheLimit: 64 * 1024 * 1024)
        let config = ModelConfiguration(id: newId)
        container = try await LLMModelFactory.shared.loadContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            configuration: config
        ) { p in
            progress(String(format: "MLX download %.0f%%", p.fractionCompleted * 100))
        }
    }

    func unload() {
        container = nil
    }

    struct RequestTiming {
        let promptTokens: Int
        let genTokens: Int
        let prefillSeconds: Double   // request start -> first chunk
        let decodeSeconds: Double    // first chunk -> last chunk
        let preview: String
    }

    func runRequest(system: String, user: String, genTokens: Int) async throws -> RequestTiming {
        guard let container else { throw EngineError(message: "MLX model not loaded") }
        let input = UserInput(chat: [.system(system), .user(user)])
        let lmInput = try await container.prepare(input: input)
        let params = GenerateParameters(maxTokens: genTokens, temperature: 0)

        let t0 = CFAbsoluteTimeGetCurrent()
        var firstChunkAt: Double? = nil
        var lastChunkAt = t0
        var preview = ""
        var promptTokens = 0
        var generated = 0

        let stream = try await container.generate(input: lmInput, parameters: params)
        for await generation in stream {
            switch generation {
            case .chunk(let text):
                let now = CFAbsoluteTimeGetCurrent()
                if firstChunkAt == nil { firstChunkAt = now }
                lastChunkAt = now
                if preview.utf8.count < 200 { preview += text }
            case .info(let info):
                promptTokens = info.promptTokenCount
                generated = info.generationTokenCount
            default:
                break
            }
        }

        let first = firstChunkAt ?? lastChunkAt
        return RequestTiming(
            promptTokens: promptTokens,
            genTokens: generated,
            prefillSeconds: first - t0,
            decodeSeconds: lastChunkAt - first,
            preview: String(preview.prefix(200))
        )
    }

    // MARK: - Batched decode (DIY continuous batch, lockstep)

    struct BatchedTiming {
        let batchSize: Int
        let promptTokensPerSeq: Int   // padded common length
        let genTokens: Int
        let prefillSeconds: Double    // one batched prefill for all B sequences
        let decodeSeconds: Double     // lockstep [B,1] steps, first -> last token
        let previews: [String]
    }

    // MARK: - vllm-swift continuous-batching engine

    struct VLLMScenarioResult {
        struct Req {
            let id: Int
            let promptTokens: Int
            let arrivalOffset: Double
            let ttft: Double
            let decodeSeconds: Double
            let genTokens: Int
            let preview: String
        }
        let reqs: [Req]
        let wall: Double
        let stepSummary: String
    }

    func runVLLMScenario(
        system: String, users: [String], genTokens: Int, maxBatch: Int,
        arrivalOffsets: [Double], progress: @escaping (String) -> Void
    ) async throws -> VLLMScenarioResult {
        guard let container else { throw EngineError(message: "MLX model not loaded") }
        return try await container.perform { (context: ModelContext) -> VLLMScenarioResult in
            var tokenLists: [[Int32]] = []
            for user in users {
                let input = UserInput(chat: [.system(system), .user(user)])
                let lmInput = try await context.processor.prepare(input: input)
                tokenLists.append(lmInput.text.tokens.asArray(Int32.self))
            }
            let padTok = context.tokenizer.encode(text: "\n").last.map(Int32.init) ?? 198
            let engine = VLLMEngine(model: context.model, padToken: padTok, maxBatch: maxBatch)
            // Shorter chained-decode graphs on iPhone: keeps peak transient
            // memory (8 chained steps of layer activations) under jetsam.
            engine.decodeChunk = 4
            engine.log = { progress($0) }

            let base = CFAbsoluteTimeGetCurrent() + 0.05
            let requests = tokenLists.enumerated().map { i, toks in
                EngineRequest(
                    id: i, promptTokens: toks, maxTokens: genTokens,
                    arrivalTime: base + arrivalOffsets[i])
            }
            let report = try engine.run(requests: requests)

            var byBatch: [Int: (n: Int, total: Double)] = [:]
            for s in report.stepStats {
                let cur = byBatch[s.batchSize] ?? (0, 0)
                byBatch[s.batchSize] = (cur.n + 1, cur.total + s.stepSeconds)
            }
            let summary = byBatch.sorted { $0.key < $1.key }.map { b, agg in
                String(format: "B%d %.0ft/s", b, Double(b) / (agg.total / Double(agg.n)))
            }.joined(separator: " ")

            let reqs = report.results.map { r in
                VLLMScenarioResult.Req(
                    id: r.id,
                    promptTokens: r.promptLength,
                    arrivalOffset: r.arrivalTime - base,
                    ttft: r.ttft,
                    decodeSeconds: r.decodeSeconds,
                    genTokens: r.tokens.count,
                    preview: String(context.tokenizer.decode(tokenIds: r.tokens.map(Int.init)).prefix(160))
                )
            }
            return VLLMScenarioResult(reqs: reqs, wall: report.wallSeconds, stepSummary: summary)
        }
    }

    /// Batched greedy generation: all prompts are padded to the SAME token
    /// count (filler newline tokens spliced in before the tail), so batched
    /// prefill needs only the standard causal mask and decode runs lockstep
    /// [B, 1] steps with no padding mask at all. This is the experiment that
    /// asks whether MLX kernels amortize weight reads across a batch.
    func runBatchedBench(
        system: String, users: [String], genTokens: Int,
        progress: @escaping (String) -> Void
    ) async throws -> BatchedTiming {
        guard let container else { throw EngineError(message: "MLX model not loaded") }

        return try await container.perform { (context: ModelContext) -> BatchedTiming in
            // Tokenize each templated prompt via the model's own processor.
            var tokenLists: [[Int32]] = []
            for user in users {
                let input = UserInput(chat: [.system(system), .user(user)])
                let lmInput = try await context.processor.prepare(input: input)
                tokenLists.append(lmInput.text.tokens.asArray(Int32.self))
            }

            // Pad every sequence to the max length by splicing newline tokens
            // in front of the last 20 tokens (inside the user message, ahead
            // of the chat-template tail so the assistant cue stays intact).
            let padToken = context.tokenizer.encode(text: "\n").last.map(Int32.init) ?? 198
            let maxLen = tokenLists.map(\.count).max() ?? 0
            let padded = tokenLists.map { toks -> [Int32] in
                let need = maxLen - toks.count
                guard need > 0 else { return toks }
                let cut = max(0, toks.count - 20)
                return Array(toks[..<cut])
                    + Array(repeating: padToken, count: need)
                    + Array(toks[cut...])
            }
            let B = padded.count

            let model = context.model
            let caches = try model.newCache(parameters: nil)
            let inputs = MLXArray(padded.flatMap { $0 }).reshaped([B, maxLen])

            // Batched prefill in chunks, evaluating ONLY cache state so the
            // lm_head output for prompt positions is never computed (lazy
            // eval). Materializing full prefill logits would be
            // [B, L, vocab] ≈ 2.6 GB at B=8/L≈1060 — instant jetsam kill.
            let t0 = CFAbsoluteTimeGetCurrent()
            let prefillChunk = 512
            let lastIndex = maxLen - 1
            var pos = 0
            while pos < lastIndex {
                let n = min(prefillChunk, lastIndex - pos)
                _ = model(inputs[0..., pos ..< (pos + n)], cache: caches)
                eval(caches.flatMap { $0.innerState() })
                pos += n
            }
            // Final prompt token -> [B, 1, vocab] logits -> first sampled token.
            var logits = model(inputs[0..., lastIndex ..< maxLen], cache: caches)
            var last = argMax(logits[0..., logits.dim(1) - 1, 0...], axis: -1)  // [B]
            eval(last)
            let prefillDone = CFAbsoluteTimeGetCurrent()

            var streams: [[Int32]] = last.asArray(Int32.self).map { [$0] }
            progress("batched prefill \(B)×\(maxLen) in \(String(format: "%.2f", prefillDone - t0))s")

            // Lockstep decode.
            for step in 1..<genTokens {
                logits = model(last.reshaped([B, 1]), cache: caches)
                last = argMax(logits[0..., logits.dim(1) - 1, 0...], axis: -1)
                eval(last)
                let toks = last.asArray(Int32.self)
                for i in 0..<B { streams[i].append(toks[i]) }
                if step % 32 == 0 { progress("decode step \(step)/\(genTokens)") }
            }
            let decodeDone = CFAbsoluteTimeGetCurrent()

            let previews = streams.map { toks in
                String(context.tokenizer.decode(tokenIds: toks.map(Int.init)).prefix(200))
            }
            return BatchedTiming(
                batchSize: B,
                promptTokensPerSeq: maxLen,
                genTokens: genTokens,
                prefillSeconds: prefillDone - t0,
                decodeSeconds: decodeDone - prefillDone,
                previews: previews
            )
        }
    }
}
