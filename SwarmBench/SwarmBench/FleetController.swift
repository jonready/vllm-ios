import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

// SwarmBench: one user request fans out to four specialist agents served
// concurrently by the VLLMEngine — batched decode, shared-prefix cache,
// streaming tokens, and early exit as each agent finishes.

@MainActor
final class FleetController: ObservableObject {

    struct ModelOption: Identifiable, Equatable {
        let id: String
        let family: String
        let variant: String
    }

    static let models: [ModelOption] = [
        .init(id: "mlx-community/Qwen3.5-0.8B-4bit", family: "Qwen 3.5", variant: "0.8B"),
        .init(id: "mlx-community/Qwen3.5-0.8B-OptiQ-4bit", family: "Qwen 3.5", variant: "OptiQ"),
        .init(id: "mlx-community/Qwen3.5-0.8B-mixed_4_6", family: "Qwen 3.5", variant: "mixed 4/6"),
        .init(id: "mlx-community/Qwen3.5-0.8B-MLX-8bit", family: "Qwen 3.5", variant: "8-bit"),
    ]

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

    struct Chat: Identifiable {
        let id = UUID()
        var turns: [Turn] = []
        var title: String { turns.first?.question ?? "New chat" }
    }

    @Published var selectedModel: ModelOption = FleetController.models[0]
    @Published var modelState: ModelState = .idle
    @Published var chats: [Chat]
    @Published var currentChatID: UUID
    @Published var isRunning = false

    // Settings
    @Published var maxTokensPerAgent: Int = 160
    @Published var prefixCachingEnabled: Bool = true
    @Published var agentCount: Int = 4  // 1...8 simultaneous agents

    private var container: ModelContainer?

    init() {
        let first = Chat()
        chats = [first]
        currentChatID = first.id
    }

    var currentChatIndex: Int {
        chats.firstIndex { $0.id == currentChatID } ?? 0
    }

    var currentChat: Chat { chats[currentChatIndex] }

    // MARK: - Chats

    func newChat() {
        guard !isRunning else { return }
        if currentChat.turns.isEmpty { return }
        let chat = Chat()
        chats.insert(chat, at: 0)
        currentChatID = chat.id
    }

    func select(chat: Chat) {
        guard !isRunning else { return }
        currentChatID = chat.id
    }

    func delete(at offsets: IndexSet) {
        guard !isRunning else { return }
        chats.remove(atOffsets: offsets)
        if chats.isEmpty { chats = [Chat()] }
        if !chats.contains(where: { $0.id == currentChatID }) {
            currentChatID = chats[0].id
        }
    }

    // MARK: - Model

    func select(model: ModelOption) {
        guard model != selectedModel, !isRunning else { return }
        selectedModel = model
        container = nil
        modelState = .idle
        loadModelIfNeeded()
    }

    func loadModelIfNeeded() {
        guard container == nil else { return }
        if case .loading = modelState { return }
        let modelId = selectedModel.id
        modelState = .loading("Preparing model…")
        Task.detached(priority: .userInitiated) {
            do {
                MLX.GPU.set(cacheLimit: 64 * 1024 * 1024)
                let loaded = try await LLMModelFactory.shared.loadContainer(
                    from: #hubDownloader(),
                    using: #huggingFaceTokenizerLoader(),
                    configuration: .init(id: modelId)
                ) { p in
                    Task { @MainActor in
                        self.modelState = .loading(String(
                            format: "Downloading… %.0f%%", p.fractionCompleted * 100))
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

    // MARK: - The swarm

    static let lenses: [(title: String, instruction: String)] = [
        ("Key facts", "State the most important concrete facts relevant to the request."),
        ("Plan", "Give a short, concrete step-by-step plan for the request."),
        ("Risks", "Point out the pitfalls, gotchas, and failure modes to watch for."),
        ("Alternatives", "Suggest different approaches or options worth considering."),
        ("Examples", "Give concrete examples or precedents that illuminate the request."),
        ("Costs", "Estimate the money, time, and effort involved."),
        ("Next steps", "Name the single best immediate next action and why."),
        ("Contrarian", "Argue the strongest case against the obvious approach."),
    ]

    static let systemPrompt = """
    You are one specialist in a team of agents answering the same user \
    request in parallel. Answer only your assigned angle. Be concise and \
    concrete: a few sentences or a short list, plain text, no preamble and no \
    mention of the team.
    """

    func send(_ question: String) {
        let q = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty, !isRunning, let container else { return }
        isRunning = true
        let activeLenses = Array(Self.lenses.prefix(max(1, min(8, agentCount))))
        let agents = activeLenses.enumerated().map { i, lens in
            AgentStream(id: i, title: lens.title)
        }
        let chatID = currentChatID
        chats[currentChatIndex].turns.append(Turn(question: q, agents: agents))
        let genTokens = maxTokensPerAgent
        let usePrefix = prefixCachingEnabled

        Task.detached(priority: .userInitiated) {
            @MainActor func update(_ mutate: (inout Turn) -> Void) {
                guard let ci = self.chats.firstIndex(where: { $0.id == chatID }),
                      !self.chats[ci].turns.isEmpty else { return }
                mutate(&self.chats[ci].turns[self.chats[ci].turns.count - 1])
            }
            do {
                let report = try await container.perform { (context: ModelContext) -> EngineRunReport in
                    var tokenLists: [[Int32]] = []
                    for lens in activeLenses {
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
                        eosTokens: eos, maxBatch: activeLenses.count)
                    engine.decodeChunk = 4

                    var prefix: PrefixCache? = nil
                    if usePrefix {
                        let shared = PrefixCache.longestCommonPrefix(of: tokenLists)
                        if !shared.isEmpty { prefix = try engine.cachePrefix(shared) }
                    }

                    let now = CFAbsoluteTimeGetCurrent()
                    let requests = tokenLists.enumerated().map { i, toks in
                        EngineRequest(id: i, promptTokens: toks, maxTokens: genTokens, arrivalTime: now)
                    }

                    // Accumulate tokens per agent; decode the running list each
                    // callback so partial UTF-8 never reaches the UI.
                    var streams: [[Int32]] = Array(repeating: [], count: requests.count)
                    return try engine.run(requests: requests, prefix: prefix) { id, newTokens, done in
                        streams[id].append(contentsOf: newTokens)
                        var text = context.tokenizer.decode(tokenIds: streams[id].map(Int.init))
                        if let e = context.tokenizer.eosToken {
                            text = text.replacingOccurrences(of: e, with: "")
                        }
                        let snapshot = text
                        Task { @MainActor in
                            update { turn in
                                turn.agents[id].text = snapshot
                                if done { turn.agents[id].done = true }
                            }
                        }
                    }
                }
                let totalGen = report.results.reduce(0) { $0 + $1.tokens.count }
                let stats = String(
                    format: "%d agents · %d tokens · %.1fs · %.0f tok/s",
                    report.results.count, totalGen, report.wallSeconds,
                    Double(totalGen) / report.wallSeconds)
                await MainActor.run {
                    update { turn in
                        for i in turn.agents.indices { turn.agents[i].done = true }
                        turn.stats = stats
                    }
                    self.isRunning = false
                }
            } catch {
                await MainActor.run {
                    update { turn in turn.stats = "failed: \(error.localizedDescription)" }
                    self.isRunning = false
                }
            }
        }
    }
}
