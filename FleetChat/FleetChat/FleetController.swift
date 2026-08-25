import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

// FleetChat: one user request fans out to four specialist agents served
// concurrently by the VLLMEngine — batched decode, shared-prefix cache,
// streaming tokens, and early exit as each agent finishes.

@MainActor
final class FleetController: ObservableObject {
    static let modelId = "mlx-community/Qwen3.5-0.8B-4bit"

    enum ModelState: Equatable {
        case idle
        case loading(String)
        case ready
        case failed(String)
    }

    struct AgentStream: Identifiable {
        let id: Int
        let title: String
        var text: String = ""
        var done: Bool = false
    }

    struct Turn: Identifiable {
        let id = UUID()
        let question: String
        var agents: [AgentStream]
        var stats: String? = nil
    }

    @Published var modelState: ModelState = .idle
    @Published var turns: [Turn] = []
    @Published var isRunning = false

    private var container: ModelContainer?

    static let lenses: [(title: String, instruction: String)] = [
        ("Key facts", "State the most important concrete facts relevant to the request."),
        ("Plan", "Give a short, concrete step-by-step plan for the request."),
        ("Risks", "Point out the pitfalls, gotchas, and failure modes to watch for."),
        ("Alternatives", "Suggest different approaches or options worth considering."),
    ]

    static let systemPrompt = """
    You are one specialist in a team of four agents answering the same user \
    request in parallel. Answer only your assigned angle. Be concise and \
    concrete: a few sentences or a short list, plain text, no preamble and no \
    mention of the team.
    """

    func loadModelIfNeeded() {
        guard container == nil, modelState != .ready else { return }
        if case .loading = modelState { return }
        modelState = .loading("Preparing model…")
        Task.detached(priority: .userInitiated) {
            do {
                MLX.GPU.set(cacheLimit: 64 * 1024 * 1024)
                let loaded = try await LLMModelFactory.shared.loadContainer(
                    from: #hubDownloader(),
                    using: #huggingFaceTokenizerLoader(),
                    configuration: .init(id: FleetController.modelId)
                ) { p in
                    Task { @MainActor in
                        self.modelState = .loading(String(
                            format: "Downloading model… %.0f%%", p.fractionCompleted * 100))
                    }
                }
                await MainActor.run {
                    self.container = loaded
                    self.modelState = .ready
                }
            } catch {
                await MainActor.run {
                    self.modelState = .failed("Model load failed: \(error.localizedDescription)")
                }
            }
        }
    }

    func send(_ question: String) {
        let q = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty, !isRunning, let container else { return }
        isRunning = true
        let agents = Self.lenses.enumerated().map { i, lens in
            AgentStream(id: i, title: lens.title)
        }
        turns.append(Turn(question: q, agents: agents))
        let turnIndex = turns.count - 1

        Task.detached(priority: .userInitiated) {
            do {
                let report = try await container.perform { (context: ModelContext) -> EngineRunReport in
                    var tokenLists: [[Int32]] = []
                    for lens in FleetController.lenses {
                        let user = "Angle: \(lens.title). \(lens.instruction)\n\nUser request: \(q)"
                        let input = UserInput(chat: [
                            .system(FleetController.systemPrompt), .user(user),
                        ])
                        let lmInput = try await context.processor.prepare(input: input)
                        tokenLists.append(lmInput.text.tokens.asArray(Int32.self))
                    }

                    let padTok = context.tokenizer.encode(text: "\n").last.map(Int32.init) ?? 198
                    var eos: Set<Int32> = []
                    if let e = context.tokenizer.eosTokenId { eos.insert(Int32(e)) }

                    let engine = VLLMEngine(
                        model: context.model, padToken: padTok,
                        eosTokens: eos, maxBatch: FleetController.lenses.count)
                    engine.decodeChunk = 4

                    var prefix: PrefixCache? = nil
                    let shared = PrefixCache.longestCommonPrefix(of: tokenLists)
                    if !shared.isEmpty { prefix = try engine.cachePrefix(shared) }

                    let now = CFAbsoluteTimeGetCurrent()
                    let requests = tokenLists.enumerated().map { i, toks in
                        EngineRequest(id: i, promptTokens: toks, maxTokens: 160, arrivalTime: now)
                    }

                    // Accumulate tokens per agent; decode the running list each
                    // callback (cheap at these lengths) so partial UTF-8 never shows.
                    var streams: [[Int32]] = Array(repeating: [], count: requests.count)
                    return try engine.run(requests: requests, prefix: prefix) { id, newTokens, done in
                        streams[id].append(contentsOf: newTokens)
                        var text = context.tokenizer.decode(tokenIds: streams[id].map(Int.init))
                        if let e = context.tokenizer.eosToken {
                            text = text.replacingOccurrences(of: e, with: "")
                        }
                        let snapshot = text
                        Task { @MainActor in
                            guard self.turns.indices.contains(turnIndex) else { return }
                            self.turns[turnIndex].agents[id].text = snapshot
                            if done { self.turns[turnIndex].agents[id].done = true }
                        }
                    }
                }
                let totalGen = report.results.reduce(0) { $0 + $1.tokens.count }
                let stats = String(
                    format: "%d agents · %d tokens · %.1fs · %.0f tok/s",
                    report.results.count, totalGen, report.wallSeconds,
                    Double(totalGen) / report.wallSeconds)
                await MainActor.run {
                    if self.turns.indices.contains(turnIndex) {
                        for i in self.turns[turnIndex].agents.indices {
                            self.turns[turnIndex].agents[i].done = true
                        }
                        self.turns[turnIndex].stats = stats
                    }
                    self.isRunning = false
                }
            } catch {
                await MainActor.run {
                    if self.turns.indices.contains(turnIndex) {
                        self.turns[turnIndex].stats = "failed: \(error.localizedDescription)"
                    }
                    self.isRunning = false
                }
            }
        }
    }
}
