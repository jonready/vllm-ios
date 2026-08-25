import Foundation
import UIKit

// One stage of a sweep: either a set of concurrency levels or a burst series,
// with a full engine config. Sweeps are arrays of stages executed in order.
struct StageConfig {
    var label: String
    var preset: PromptPreset = .medium
    var genTokens: Int = 128
    var requestCount: Int = 8
    var concurrencies: [Int] = [1]      // used when burst == false
    var burst: Bool = false
    var burstCount: Int = 3
    var burstGapSeconds: Int = 30
    var kvUnified = false
    var flashAttention = false
    var nUbatch = 512
    var thermalGate = true              // cool to nominal/fair before each level / first burst
    // Mandatory idle seconds between concurrency-level runs. Needed because the
    // die throttles below thermalState's granularity (esp. under an ice pack,
    // where reported state stays "nominal" while prefill decays run over run).
    var interRunGapSeconds = 0
}

@MainActor
final class BenchController: ObservableObject {
    enum State: Equatable {
        case idle
        case loadingModel
        case running(String)
    }

    @Published var state: State = .idle
    @Published var loadedModelName: String? = nil
    @Published var modelInfo: String = ""
    @Published var log: [String] = []
    @Published var runs: [BenchRunResult] = []

    // Manual config
    @Published var preset: PromptPreset = .medium
    @Published var genTokens: Int = 128
    @Published var requestCount: Int = 8
    @Published var concurrencyLevels: Set<Int> = [1, 2, 4]
    @Published var kvUnified: Bool = false
    @Published var flashAttention: Bool = false
    @Published var thermalGate: Bool = true
    @Published var burstMode: Bool = false
    @Published var burstCount: Int = 3
    @Published var burstGapSeconds: Int = 30
    @Published var nUbatch: Int = 512

    private final class CancelToken: @unchecked Sendable { var cancelled = false }
    private let cancelToken = CancelToken()

    private static var resultsURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("bench_results.json")
    }

    init() {
        loadResults()
    }

    var isBusy: Bool { state != .idle }

    func appendLog(_ line: String) {
        log.append(line)
        if log.count > 300 { log.removeFirst(log.count - 300) }
    }

    // MARK: - Model load

    func loadModel(at url: URL) {
        guard !isBusy else { return }
        state = .loadingModel
        appendLog("Loading \(url.lastPathComponent)…")
        Task.detached(priority: .userInitiated) {
            do {
                let t0 = CFAbsoluteTimeGetCurrent()
                try LlamaEngine.shared.loadModel(path: url.path)
                let dt = CFAbsoluteTimeGetCurrent() - t0
                let engine = LlamaEngine.shared
                let info = String(
                    format: "%.0fM params, %.0f MB, ctx_train %d, loaded in %.1fs",
                    Double(engine.modelParams) / 1e6,
                    Double(engine.modelSizeBytes) / 1_048_576,
                    engine.ctxTrain, dt
                )
                await MainActor.run {
                    self.loadedModelName = engine.modelName
                    self.modelInfo = info
                    self.appendLog("Loaded: \(info)")
                    self.state = .idle
                }
            } catch {
                await MainActor.run {
                    self.appendLog("Load failed: \(error)")
                    self.state = .idle
                }
            }
        }
    }

    // MARK: - Sweep plans

    /// The clean scaling curve: base config (winner of the config experiments),
    /// c1/2/4/8 with a mandatory 90s gap between runs so invisible die-level
    /// throttling can't contaminate later levels. Run with the phone ON the
    /// ice pack. (The 2026-08-24 config sweep proved back-to-back runs decay
    /// even at reported "nominal": prefill fell 1034→648 t/s across one stage.)
    static func scalingSweepPlan() -> [StageConfig] {
        [
            StageConfig(label: "scale-base", genTokens: 128, concurrencies: [1, 2, 4, 8],
                        interRunGapSeconds: 90),
        ]
    }

    /// Finds the sustainable burst cadence: c8 bursts in agent shape (gen 64),
    /// gap 0 (degradation floor), then 30s and 90s gaps. Run with the phone
    /// OFF the ice pack, sitting on a table — this measures real-world recovery.
    static func dutySweepPlan() -> [StageConfig] {
        [
            StageConfig(label: "duty-gap0", genTokens: 64, burst: true,
                        burstCount: 3, burstGapSeconds: 0),
            StageConfig(label: "duty-gap30", genTokens: 64, burst: true,
                        burstCount: 4, burstGapSeconds: 30),
            StageConfig(label: "duty-gap90", genTokens: 64, burst: true,
                        burstCount: 4, burstGapSeconds: 90),
        ]
    }

    func startScalingSweep() { run(stages: Self.scalingSweepPlan(), session: "scaling sweep") }
    func startDutySweep() { run(stages: Self.dutySweepPlan(), session: "duty-cycle sweep") }

    // MARK: - MLX arm (single stream — MLX has no batched decode)

    @Published var mlxLoadedId: String? = nil

    func loadMLXModel(id: String) {
        guard !isBusy else { return }
        state = .running("MLX: loading \(id)…")
        appendLog("Loading MLX model \(id)…")
        Task.detached(priority: .userInitiated) {
            defer { Task { @MainActor in self.state = .idle } }
            do {
                let t0 = CFAbsoluteTimeGetCurrent()
                try await MLXEngine.shared.load(modelId: id) { msg in
                    Task { @MainActor in self.state = .running(msg) }
                }
                let dt = CFAbsoluteTimeGetCurrent() - t0
                await MainActor.run {
                    self.mlxLoadedId = id
                    self.appendLog(String(format: "MLX model loaded in %.1fs", dt))
                }
            } catch {
                await MainActor.run { self.appendLog("MLX load failed: \(error)") }
            }
        }
    }

    func startMLXBatchedBench() {
        guard !isBusy, let modelId = mlxLoadedId else { return }
        let variant = modelId.split(separator: "/").last.map {
            $0.replacingOccurrences(of: "Qwen3.5-0.8B-", with: "")
        } ?? "?"
        let batch = 8
        let genTokens = 128
        state = .running("MLX batched: preparing…")
        appendLog("=== MLX BATCHED arm: \(modelId), B=\(batch), medium, gen \(genTokens) ===")
        UIApplication.shared.isIdleTimerDisabled = true

        Task.detached(priority: .userInitiated) {
            defer {
                Task { @MainActor in
                    UIApplication.shared.isIdleTimerDisabled = false
                    self.state = .idle
                }
            }
            do {
                try await MLXEngine.shared.load(modelId: modelId) { msg in
                    Task { @MainActor in self.state = .running(msg) }
                }
                // Untimed warmup at the same batch size (kernel compilation
                // for the [B, ...] shapes).
                _ = try await MLXEngine.shared.runBatchedBench(
                    system: "You are helpful.",
                    users: Array(repeating: "Say OK.", count: batch),
                    genTokens: 4, progress: { _ in })
                await MainActor.run { self.appendLog("Batched warmup done") }

                let prompts = Workloads.buildUserPrompts(preset: .medium, count: batch)
                let thermalStart = ProcessInfo.processInfo.thermalState.label
                let t0 = CFAbsoluteTimeGetCurrent()
                let r = try await MLXEngine.shared.runBatchedBench(
                    system: Workloads.systemPrompt, users: prompts, genTokens: genTokens,
                    progress: { msg in
                        Task { @MainActor in self.state = .running("MLX batched: \(msg)") }
                    })
                let wall = CFAbsoluteTimeGetCurrent() - t0

                // Shared batched prefill: split its time evenly across requests
                // so the summed aggregates stay correct.
                let reqResults = (0..<r.batchSize).map { i in
                    BenchRequestResult(
                        index: i,
                        promptTokens: r.promptTokensPerSeq,
                        genTokens: r.genTokens,
                        prefillSeconds: r.prefillSeconds / Double(r.batchSize),
                        ttftSeconds: r.prefillSeconds,
                        decodeSeconds: r.decodeSeconds,
                        outputPreview: r.previews[i]
                    )
                }
                var result = BenchRunResult(
                    timestamp: Date(),
                    modelName: modelId,
                    concurrency: r.batchSize,
                    requestCount: r.batchSize,
                    genTokensPerRequest: genTokens,
                    wallSeconds: wall,
                    requests: reqResults,
                    nCtx: r.promptTokensPerSeq + genTokens,
                    nThreads: 0,
                    availableMemBeforeMB: 0,
                    availableMemAfterMB: 0,
                    thermalState: ProcessInfo.processInfo.thermalState.label,
                    thermalStateStart: thermalStart,
                    kvUnified: nil,
                    flashAttention: nil
                )
                result.stageLabel = "mlxbatch\(r.batchSize)-\(variant)"
                result.engine = "mlx"
                let res = result
                await MainActor.run {
                    self.runs.insert(res, at: 0)
                    self.saveResults()
                    let perStream = Double(res.genTokensPerRequest - 1) / r.decodeSeconds
                    self.appendLog(String(
                        format: "mlxbatch%d-%@: prefill %.0f t/s | dec %.1f t/s/stream, %.1f t/s AGG | wall %.1fs | %@→%@",
                        r.batchSize, variant,
                        Double(r.batchSize * r.promptTokensPerSeq) / r.prefillSeconds,
                        perStream, perStream * Double(r.batchSize),
                        wall, res.thermalStateStart ?? "?", res.thermalState))
                    self.appendLog("=== MLX batched arm complete ===")
                }
            } catch {
                await MainActor.run { self.appendLog("MLX batched arm stopped: \(error)") }
            }
        }
    }

    func startVLLMBench() {
        guard !isBusy, let modelId = mlxLoadedId else { return }
        let variant = modelId.split(separator: "/").last.map {
            $0.replacingOccurrences(of: "Qwen3.5-0.8B-", with: "")
        } ?? "?"
        let n = 16
        // Batch cap follows the highest selected concurrency level (1–8).
        let maxBatch = min(8, max(1, concurrencyLevels.max() ?? 8))
        let genTokens = 128
        state = .running("vllm-swift: preparing…")
        appendLog("=== vllm-swift engine: \(modelId), n=\(n), maxBatch=\(maxBatch), gen \(genTokens) ===")
        UIApplication.shared.isIdleTimerDisabled = true

        Task.detached(priority: .userInitiated) {
            defer {
                Task { @MainActor in
                    UIApplication.shared.isIdleTimerDisabled = false
                    self.state = .idle
                }
            }
            do {
                try await MLXEngine.shared.load(modelId: modelId) { msg in
                    Task { @MainActor in self.state = .running(msg) }
                }
                // Warmup (kernel compilation) — tiny 2-request burst.
                _ = try await MLXEngine.shared.runVLLMScenario(
                    system: "You are helpful.",
                    users: ["Say OK.", "Say OK."],
                    genTokens: 4, maxBatch: 2,
                    arrivalOffsets: [0, 0], progress: { _ in })
                await MainActor.run { self.appendLog("vllm warmup done") }

                let prompts = Workloads.buildUserPrompts(preset: .medium, count: n)
                let scenarios: [(String, [Double])] = [
                    ("vllm-burst", Array(repeating: 0, count: n)),
                    ("vllm-stagger", (0..<n).map { Double($0) * 0.5 }),
                ]
                await MainActor.run { self.appendLog("engine batch cap: \(maxBatch)") }
                for (label, arrivals) in scenarios {
                    if self.cancelToken.cancelled { break }
                    // Cool between scenarios so they're comparable.
                    var waited = 0
                    while ProcessInfo.processInfo.thermalState != .nominal
                        && ProcessInfo.processInfo.thermalState != .fair && waited < 180 {
                        await MainActor.run { self.state = .running("\(label): cooling (\(waited)s)…") }
                        try await Task.sleep(nanoseconds: 5_000_000_000)
                        waited += 5
                    }
                    let thermalStart = ProcessInfo.processInfo.thermalState.label
                    await MainActor.run { self.state = .running("\(label) running…") }
                    let r = try await MLXEngine.shared.runVLLMScenario(
                        system: Workloads.systemPrompt, users: prompts,
                        genTokens: genTokens, maxBatch: maxBatch,
                        arrivalOffsets: arrivals,
                        progress: { msg in
                            Task { @MainActor in self.state = .running("\(label): \(msg)") }
                        })
                    // prefillSeconds stays 0 for engine runs: prefill and decode
                    // interleave under continuous batching, so the only honest
                    // aggregate is genTokens / wall (which the shared metrics
                    // then compute, since wall - Σprefill == wall).
                    let reqResults = r.reqs.map { q in
                        BenchRequestResult(
                            index: q.id,
                            promptTokens: q.promptTokens,
                            genTokens: q.genTokens,
                            prefillSeconds: 0,
                            ttftSeconds: q.ttft,
                            decodeSeconds: q.decodeSeconds,
                            outputPreview: q.preview)
                    }
                    var result = BenchRunResult(
                        timestamp: Date(),
                        modelName: modelId,
                        concurrency: maxBatch,
                        requestCount: n,
                        genTokensPerRequest: genTokens,
                        wallSeconds: r.wall,
                        requests: reqResults,
                        nCtx: 0, nThreads: 0,
                        availableMemBeforeMB: 0, availableMemAfterMB: 0,
                        thermalState: ProcessInfo.processInfo.thermalState.label,
                        thermalStateStart: thermalStart,
                        kvUnified: nil, flashAttention: nil)
                    result.stageLabel = "\(label)-b\(maxBatch)-\(variant)"
                    result.engine = "vllm-swift"
                    let res = result
                    let summary = r.stepSummary
                    await MainActor.run {
                        self.runs.insert(res, at: 0)
                        self.saveResults()
                        let totalGen = res.requests.reduce(0) { $0 + $1.genTokens }
                        let ttfts = res.requests.map(\.ttftSeconds).sorted()
                        let aggStr = String(format: "%.1f", Double(totalGen) / res.wallSeconds)
                        let p50Str = String(format: "%.2f", ttfts[ttfts.count / 2])
                        let maxStr = String(format: "%.2f", ttfts.last ?? 0)
                        let wallStr = String(format: "%.1f", res.wallSeconds)
                        self.appendLog("\(res.stageLabel ?? ""): wall \(wallStr)s | agg gen \(aggStr) t/s | ttft p50 \(p50Str)s max \(maxStr)s | \(summary) | \(res.thermalStateStart ?? "?")→\(res.thermalState)")
                    }
                }
                await MainActor.run { self.appendLog("=== vllm-swift bench complete ===") }
            } catch {
                await MainActor.run { self.appendLog("vllm-swift bench stopped: \(error)") }
            }
        }
    }

    /// MLX mirror of the llama.cpp scaling sweep: the vllm-swift engine runs
    /// an 8-request burst at batch caps 1/2/4/8, thermal-gated with mandatory
    /// 90s gaps between levels — directly comparable, level for level.
    func startMLXScalingSweep() {
        guard !isBusy, let modelId = mlxLoadedId else { return }
        let variant = modelId.split(separator: "/").last.map {
            $0.replacingOccurrences(of: "Qwen3.5-0.8B-", with: "")
        } ?? "?"
        let nReq = 8
        let genTokens = 128
        cancelToken.cancelled = false
        let token = cancelToken
        state = .running("MLX scaling sweep: preparing…")
        appendLog("=== MLX scaling sweep: \(modelId), b1/2/4/8, \(nReq) req, gen \(genTokens) ===")
        UIApplication.shared.isIdleTimerDisabled = true

        Task.detached(priority: .userInitiated) {
            defer {
                Task { @MainActor in
                    UIApplication.shared.isIdleTimerDisabled = false
                    self.state = .idle
                }
            }
            do {
                try await MLXEngine.shared.load(modelId: modelId) { msg in
                    Task { @MainActor in self.state = .running(msg) }
                }
                _ = try await MLXEngine.shared.runVLLMScenario(
                    system: "You are helpful.", users: ["Say OK.", "Say OK."],
                    genTokens: 4, maxBatch: 2, arrivalOffsets: [0, 0], progress: { _ in })
                await MainActor.run { self.appendLog("warmup done") }

                let prompts = Workloads.buildUserPrompts(preset: .medium, count: nReq)
                for (i, b) in [1, 2, 4, 8].enumerated() {
                    if token.cancelled { break }
                    if i > 0 {
                        for s in stride(from: 0, to: 90, by: 5) {
                            if token.cancelled { break }
                            await MainActor.run { self.state = .running("gap before b\(b) (\(s)/90s)…") }
                            try await Task.sleep(nanoseconds: 5_000_000_000)
                        }
                    }
                    var waited = 0
                    while ProcessInfo.processInfo.thermalState != .nominal
                        && ProcessInfo.processInfo.thermalState != .fair
                        && waited < 300 && !token.cancelled {
                        await MainActor.run { self.state = .running("b\(b): cooling (\(waited)s)…") }
                        try await Task.sleep(nanoseconds: 5_000_000_000)
                        waited += 5
                    }
                    if token.cancelled { break }
                    let thermalStart = ProcessInfo.processInfo.thermalState.label
                    await MainActor.run { self.state = .running("b\(b) running…") }
                    let r = try await MLXEngine.shared.runVLLMScenario(
                        system: Workloads.systemPrompt, users: prompts,
                        genTokens: genTokens, maxBatch: b,
                        arrivalOffsets: Array(repeating: 0, count: nReq),
                        progress: { msg in
                            Task { @MainActor in self.state = .running("b\(b): \(msg)") }
                        })
                    let reqResults = r.reqs.map { q in
                        BenchRequestResult(
                            index: q.id, promptTokens: q.promptTokens, genTokens: q.genTokens,
                            prefillSeconds: 0, ttftSeconds: q.ttft,
                            decodeSeconds: q.decodeSeconds, outputPreview: q.preview)
                    }
                    var result = BenchRunResult(
                        timestamp: Date(), modelName: modelId,
                        concurrency: b, requestCount: nReq,
                        genTokensPerRequest: genTokens, wallSeconds: r.wall,
                        requests: reqResults, nCtx: 0, nThreads: 0,
                        availableMemBeforeMB: 0, availableMemAfterMB: 0,
                        thermalState: ProcessInfo.processInfo.thermalState.label,
                        thermalStateStart: thermalStart,
                        kvUnified: nil, flashAttention: nil)
                    result.stageLabel = "mlxscale-b\(b)-\(variant)"
                    result.engine = "vllm-swift"
                    let res = result
                    let summary = r.stepSummary
                    await MainActor.run {
                        self.runs.insert(res, at: 0)
                        self.saveResults()
                        let wallStr = String(format: "%.1f", res.wallSeconds)
                        self.appendLog("mlxscale-b\(b): wall \(wallStr)s | \(summary) | \(res.thermalStateStart ?? "?")→\(res.thermalState)")
                    }
                }
                await MainActor.run { self.appendLog("=== MLX scaling sweep complete ===") }
            } catch {
                await MainActor.run { self.appendLog("MLX scaling sweep stopped: \(error)") }
            }
        }
    }

    func startMLXBench() {
        guard !isBusy, let modelId = mlxLoadedId else { return }
        cancelToken.cancelled = false
        let token = cancelToken
        // e.g. "mlx-community/Qwen3.5-0.8B-OptiQ-4bit" -> "mlx-OptiQ-4bit"
        let variant = modelId.split(separator: "/").last.map {
            $0.replacingOccurrences(of: "Qwen3.5-0.8B-", with: "")
        } ?? "?"
        state = .running("MLX: preparing…")
        appendLog("=== MLX arm: \(modelId), 8 req sequential, medium, gen 128 ===")
        UIApplication.shared.isIdleTimerDisabled = true

        Task.detached(priority: .userInitiated) {
            defer {
                Task { @MainActor in
                    UIApplication.shared.isIdleTimerDisabled = false
                    self.state = .idle
                }
            }
            do {
                try await MLXEngine.shared.load(modelId: modelId) { msg in
                    Task { @MainActor in self.state = .running(msg) }
                }
                await MainActor.run { self.appendLog("MLX model loaded") }

                // Untimed warmup (Metal kernel compilation).
                _ = try await MLXEngine.shared.runRequest(
                    system: "You are helpful.", user: "Say OK.", genTokens: 8)
                await MainActor.run { self.appendLog("MLX warmup done") }

                let genTokens = 128
                let prompts = Workloads.buildUserPrompts(preset: .medium, count: 8)
                let thermalStart = ProcessInfo.processInfo.thermalState.label
                var reqResults: [BenchRequestResult] = []
                let t0 = CFAbsoluteTimeGetCurrent()

                for (i, p) in prompts.enumerated() {
                    if token.cancelled { break }
                    await MainActor.run { self.state = .running("MLX req \(i + 1)/\(prompts.count)…") }
                    let reqStart = CFAbsoluteTimeGetCurrent()
                    let r = try await MLXEngine.shared.runRequest(
                        system: Workloads.systemPrompt, user: p, genTokens: genTokens)
                    reqResults.append(BenchRequestResult(
                        index: i,
                        promptTokens: r.promptTokens,
                        genTokens: r.genTokens,
                        prefillSeconds: r.prefillSeconds,
                        ttftSeconds: (reqStart + r.prefillSeconds) - t0,
                        decodeSeconds: r.decodeSeconds,
                        outputPreview: r.preview
                    ))
                }
                let wall = CFAbsoluteTimeGetCurrent() - t0

                var result = BenchRunResult(
                    timestamp: Date(),
                    modelName: MLXEngine.shared.modelId,
                    concurrency: 1,
                    requestCount: prompts.count,
                    genTokensPerRequest: genTokens,
                    wallSeconds: wall,
                    requests: reqResults,
                    nCtx: 0,
                    nThreads: 0,
                    availableMemBeforeMB: 0,
                    availableMemAfterMB: 0,
                    thermalState: ProcessInfo.processInfo.thermalState.label,
                    thermalStateStart: thermalStart,
                    kvUnified: nil,
                    flashAttention: nil
                )
                result.stageLabel = "mlx-\(variant)"
                result.engine = "mlx"
                let r = result
                await MainActor.run {
                    self.runs.insert(r, at: 0)
                    self.saveResults()
                    self.appendLog(String(
                        format: "mlx-%@ c1: TTFT p50 %.2fs | prefill %.0f t/s | dec %.1f t/s | wall %.1fs | %@→%@",
                        variant, r.ttftP50, r.aggregatePrefillTPS, r.meanPerStreamDecodeTPS,
                        r.wallSeconds, r.thermalStateStart ?? "?", r.thermalState))
                    self.appendLog("=== MLX arm complete ===")
                }
            } catch {
                await MainActor.run { self.appendLog("MLX arm stopped: \(error)") }
            }
        }
    }

    // MARK: - Manual run (single stage built from the UI config)

    func runBenchmarks() {
        let stage = StageConfig(
            label: "manual",
            preset: preset,
            genTokens: genTokens,
            requestCount: requestCount,
            concurrencies: concurrencyLevels.sorted(),
            burst: burstMode,
            burstCount: burstCount,
            burstGapSeconds: burstGapSeconds,
            kvUnified: kvUnified,
            flashAttention: flashAttention,
            nUbatch: nUbatch,
            thermalGate: thermalGate
        )
        run(stages: [stage], session: "manual bench")
    }

    // MARK: - Executor

    private func run(stages: [StageConfig], session: String) {
        guard !isBusy, loadedModelName != nil, !stages.isEmpty else { return }
        cancelToken.cancelled = false
        let token = cancelToken
        state = .running("Preparing…")
        appendLog("=== \(session): \(stages.map(\.label).joined(separator: " → ")) ===")
        UIApplication.shared.isIdleTimerDisabled = true

        Task.detached(priority: .userInitiated) {
            defer {
                Task { @MainActor in
                    UIApplication.shared.isIdleTimerDisabled = false
                    self.state = .idle
                }
            }
            let engine = LlamaEngine.shared
            do {
                var warmedUp = false
                for stage in stages {
                    if token.cancelled { break }
                    await MainActor.run { self.appendLog("--- stage \(stage.label) ---") }

                    let userPrompts = Workloads.buildUserPrompts(preset: stage.preset, count: stage.requestCount)
                    var tokenized: [[Int32]] = []
                    for p in userPrompts {
                        let templated = engine.applyChatTemplate(system: Workloads.systemPrompt, user: p)
                        tokenized.append(try engine.tokenize(templated, addSpecial: false))
                    }
                    let sizes = tokenized.map(\.count)
                    await MainActor.run {
                        self.appendLog("\(stage.label): \(stage.requestCount) prompts, \(sizes.min() ?? 0)–\(sizes.max() ?? 0) tok")
                    }

                    if !warmedUp {
                        let warm = try engine.tokenize(
                            engine.applyChatTemplate(system: "You are helpful.", user: "Say OK."),
                            addSpecial: false)
                        _ = try engine.runBenchmark(
                            prompts: [warm], concurrency: 1, genTokens: 8,
                            kvUnified: stage.kvUnified, flashAttention: stage.flashAttention,
                            nUbatch: stage.nUbatch,
                            isCancelled: { false }, progress: { _ in })
                        warmedUp = true
                        await MainActor.run { self.appendLog("Warmup done") }
                    }

                    if stage.burst {
                        try await self.runBurstStage(stage, tokenized: tokenized, engine: engine, token: token)
                    } else {
                        try await self.runLevelsStage(stage, tokenized: tokenized, engine: engine, token: token)
                    }
                }
                await MainActor.run {
                    self.appendLog(token.cancelled ? "=== \(session) cancelled ===" : "=== \(session) complete ===")
                }
            } catch {
                await MainActor.run { self.appendLog("\(session) stopped: \(error)") }
            }
        }
    }

    private nonisolated func waitForCool(label: String, token: CancelToken) async throws {
        var waited = 0
        while ProcessInfo.processInfo.thermalState != .nominal
            && ProcessInfo.processInfo.thermalState != .fair
            && waited < 300 && !token.cancelled {
            await MainActor.run {
                self.state = .running("\(label): cooling (\(ProcessInfo.processInfo.thermalState.label), \(waited)s)…")
            }
            try await Task.sleep(nanoseconds: 5_000_000_000)
            waited += 5
        }
        if waited >= 300 {
            await MainActor.run {
                self.appendLog("⚠️ still \(ProcessInfo.processInfo.thermalState.label) after 300s — running anyway")
            }
        }
    }

    private nonisolated func runLevelsStage(
        _ stage: StageConfig, tokenized: [[Int32]], engine: LlamaEngine, token: CancelToken
    ) async throws {
        for (i, level) in stage.concurrencies.enumerated() {
            if token.cancelled { break }
            if i > 0 && stage.interRunGapSeconds > 0 {
                for s in stride(from: 0, to: stage.interRunGapSeconds, by: 2) {
                    if token.cancelled { break }
                    await MainActor.run {
                        self.state = .running("\(stage.label): gap before c\(level) (\(s)/\(stage.interRunGapSeconds)s)…")
                    }
                    try await Task.sleep(nanoseconds: 2_000_000_000)
                }
            }
            if stage.thermalGate {
                try await self.waitForCool(label: "\(stage.label) c\(level)", token: token)
            }
            if token.cancelled { break }
            await MainActor.run {
                self.state = .running("\(stage.label) c\(level) (thermal \(ProcessInfo.processInfo.thermalState.label))…")
            }
            var result = try engine.runBenchmark(
                prompts: tokenized,
                concurrency: level,
                genTokens: stage.genTokens,
                kvUnified: stage.kvUnified,
                flashAttention: stage.flashAttention,
                nUbatch: stage.nUbatch,
                isCancelled: { token.cancelled },
                progress: { msg in
                    Task { @MainActor in self.state = .running("\(stage.label) c\(level): \(msg)") }
                }
            )
            result.stageLabel = stage.label
            let r = result
            await MainActor.run {
                self.runs.insert(r, at: 0)
                self.saveResults()
                self.appendLog(String(
                    format: "%@ c%d: TTFT p50 %.2fs | prefill %.0f t/s | dec %.1f/str %.1f agg | wall %.1fs | %@→%@",
                    stage.label, level, r.ttftP50, r.aggregatePrefillTPS,
                    r.meanPerStreamDecodeTPS, r.aggregateDecodeTPS, r.wallSeconds,
                    r.thermalStateStart ?? "?", r.thermalState))
            }
        }
    }

    private nonisolated func runBurstStage(
        _ stage: StageConfig, tokenized: [[Int32]], engine: LlamaEngine, token: CancelToken
    ) async throws {
        if stage.thermalGate {
            try await self.waitForCool(label: stage.label, token: token)
        }
        for b in 1...stage.burstCount {
            if token.cancelled { break }
            let startThermal = ProcessInfo.processInfo.thermalState
            await MainActor.run {
                self.state = .running("\(stage.label) burst \(b)/\(stage.burstCount) (thermal \(startThermal.label))…")
            }
            var result = try engine.runBenchmark(
                prompts: tokenized,
                concurrency: tokenized.count,
                genTokens: stage.genTokens,
                kvUnified: stage.kvUnified,
                flashAttention: stage.flashAttention,
                nUbatch: stage.nUbatch,
                isCancelled: { token.cancelled },
                progress: { msg in
                    Task { @MainActor in self.state = .running("\(stage.label) burst \(b): \(msg)") }
                }
            )
            // Gap = at least burstGapSeconds (the die throttles below
            // thermalState's granularity), plus thermal-state recovery (cap 300s).
            let recovStart = CFAbsoluteTimeGetCurrent()
            if b < stage.burstCount {
                while !token.cancelled {
                    let elapsed = CFAbsoluteTimeGetCurrent() - recovStart
                    let thermalPending = ProcessInfo.processInfo.thermalState.rawValue > startThermal.rawValue && elapsed < 300
                    let gapPending = elapsed < Double(stage.burstGapSeconds)
                    if !thermalPending && !gapPending { break }
                    await MainActor.run {
                        self.state = .running("\(stage.label) burst \(b) gap: \(ProcessInfo.processInfo.thermalState.label), \(Int(elapsed))s…")
                    }
                    try await Task.sleep(nanoseconds: 2_000_000_000)
                }
            }
            result.recoverySeconds = CFAbsoluteTimeGetCurrent() - recovStart
            result.burstIndex = b
            result.stageLabel = stage.label
            let r = result
            await MainActor.run {
                self.runs.insert(r, at: 0)
                self.saveResults()
                let duty = r.wallSeconds / (r.wallSeconds + (r.recoverySeconds ?? 0)) * 100
                self.appendLog(String(
                    format: "%@ burst %d: wall %.1fs, %d tok | dec agg %.1f t/s | gap %.0fs | duty %.0f%% | %@→%@",
                    stage.label, b, r.wallSeconds, r.totalPromptTokens + r.totalGenTokens,
                    r.aggregateDecodeTPS, r.recoverySeconds ?? 0, duty,
                    r.thermalStateStart ?? "?", r.thermalState))
            }
        }
    }

    func cancel() {
        cancelToken.cancelled = true
    }

    // MARK: - Persistence / export

    func saveResults() {
        let session = BenchSession(
            device: DeviceInfo.machine,
            os: UIDevice.current.systemName + " " + UIDevice.current.systemVersion,
            llamaBuild: "b10612",
            runs: runs
        )
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        enc.dateEncodingStrategy = .iso8601
        if let data = try? enc.encode(session) {
            try? data.write(to: Self.resultsURL)
        }
    }

    func loadResults() {
        guard let data = try? Data(contentsOf: Self.resultsURL) else { return }
        let dec = JSONDecoder()
        dec.dateDecodingStrategy = .iso8601
        if let session = try? dec.decode(BenchSession.self, from: data) {
            runs = session.runs
        }
    }

    func clearResults() {
        runs = []
        try? FileManager.default.removeItem(at: Self.resultsURL)
    }

    var exportURL: URL? {
        FileManager.default.fileExists(atPath: Self.resultsURL.path) ? Self.resultsURL : nil
    }
}
