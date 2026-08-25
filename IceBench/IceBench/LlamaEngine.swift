import Foundation
import llama
import os

struct EngineError: Error, CustomStringConvertible {
    let message: String
    var description: String { message }
}

// All llama.cpp calls must happen on one thread; BenchController runs everything
// on a single detached task.
final class LlamaEngine {
    static let shared = LlamaEngine()

    private(set) var model: OpaquePointer?
    private(set) var vocab: OpaquePointer?
    private(set) var modelName: String = ""
    private(set) var modelSizeBytes: UInt64 = 0
    private(set) var modelParams: UInt64 = 0
    private(set) var ctxTrain: Int32 = 0

    private init() {
        llama_backend_init()
    }

    var isLoaded: Bool { model != nil }

    func loadModel(path: String) throws {
        unloadModel()
        var mp = llama_model_default_params()
        mp.n_gpu_layers = 999
        guard let m = llama_model_load_from_file(path, mp) else {
            throw EngineError(message: "Failed to load model: \((path as NSString).lastPathComponent)")
        }
        model = m
        vocab = llama_model_get_vocab(m)
        modelName = (path as NSString).lastPathComponent
        modelSizeBytes = llama_model_size(m)
        modelParams = llama_model_n_params(m)
        ctxTrain = llama_model_n_ctx_train(m)
    }

    func unloadModel() {
        if let m = model { llama_model_free(m) }
        model = nil
        vocab = nil
        modelName = ""
    }

    // MARK: - Tokenization

    func tokenize(_ text: String, addSpecial: Bool, parseSpecial: Bool = true) throws -> [llama_token] {
        guard let vocab else { throw EngineError(message: "No model loaded") }
        let utf8 = Array(text.utf8)
        var count = Int32(utf8.count) + 16
        var tokens = [llama_token](repeating: 0, count: Int(count))
        let n = utf8.withUnsafeBufferPointer { buf -> Int32 in
            buf.baseAddress!.withMemoryRebound(to: CChar.self, capacity: utf8.count) { cstr in
                llama_tokenize(vocab, cstr, Int32(utf8.count), &tokens, count, addSpecial, parseSpecial)
            }
        }
        if n < 0 {
            count = -n
            tokens = [llama_token](repeating: 0, count: Int(count))
            let n2 = utf8.withUnsafeBufferPointer { buf -> Int32 in
                buf.baseAddress!.withMemoryRebound(to: CChar.self, capacity: utf8.count) { cstr in
                    llama_tokenize(vocab, cstr, Int32(utf8.count), &tokens, count, addSpecial, parseSpecial)
                }
            }
            guard n2 >= 0 else { throw EngineError(message: "Tokenization failed (\(n2))") }
            return Array(tokens[0..<Int(n2)])
        }
        return Array(tokens[0..<Int(n)])
    }

    func piece(for token: llama_token) -> String {
        guard let vocab else { return "" }
        var buf = [CChar](repeating: 0, count: 128)
        let n = llama_token_to_piece(vocab, token, &buf, 128, 0, false)
        guard n > 0 else { return "" }
        return String(decoding: buf[0..<Int(n)].map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    // MARK: - Chat template

    func applyChatTemplate(system: String, user: String) -> String {
        guard let model else { return user }
        let tmpl = llama_model_chat_template(model, nil)
        let sysC = strdup(system)!
        let userC = strdup(user)!
        let roleSys = strdup("system")!
        let roleUser = strdup("user")!
        defer { free(sysC); free(userC); free(roleSys); free(roleUser) }
        var msgs = [
            llama_chat_message(role: UnsafePointer(roleSys), content: UnsafePointer(sysC)),
            llama_chat_message(role: UnsafePointer(roleUser), content: UnsafePointer(userC)),
        ]
        var size = Int32(2 * (system.utf8.count + user.utf8.count) + 512)
        var buf = [CChar](repeating: 0, count: Int(size))
        var n = llama_chat_apply_template(tmpl, &msgs, 2, true, &buf, size)
        if n > size {
            size = n
            buf = [CChar](repeating: 0, count: Int(size))
            n = llama_chat_apply_template(tmpl, &msgs, 2, true, &buf, size)
        }
        if n <= 0 {
            // Fallback: ChatML (Qwen family)
            return "<|im_start|>system\n\(system)<|im_end|>\n<|im_start|>user\n\(user)<|im_end|>\n<|im_start|>assistant\n"
        }
        return String(decoding: buf[0..<Int(n)].map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    // MARK: - Benchmark

    /// Runs one benchmark level: `prompts.count` requests, all submitted at t=0,
    /// executed with up to `concurrency` parallel sequences in a single shared
    /// context. Prefill is sequential per admitted request (chunks of n_batch);
    /// decode is batched across all active sequences, one token each per
    /// llama_decode. Generation is forced to exactly `genTokens` tokens per
    /// request (EOG ignored) so token counts are comparable across runs.
    func runBenchmark(
        prompts: [[llama_token]],
        concurrency: Int,
        genTokens: Int,
        kvUnified: Bool = false,
        flashAttention: Bool = false,
        nUbatch: Int = 512,
        isCancelled: () -> Bool,
        progress: (String) -> Void
    ) throws -> BenchRunResult {
        guard let model else { throw EngineError(message: "No model loaded") }

        let nSeq = concurrency
        let maxPrompt = prompts.map(\.count).max() ?? 0
        let perSeqCtx = maxPrompt + genTokens + 64
        let nBatch: Int32 = 2048
        let thermalStart = ProcessInfo.processInfo.thermalState.label

        var cp = llama_context_default_params()
        cp.n_ctx = UInt32(perSeqCtx * nSeq)
        cp.n_seq_max = UInt32(nSeq)
        cp.n_batch = UInt32(nBatch)
        cp.n_ubatch = UInt32(nUbatch)
        cp.kv_unified = kvUnified
        cp.flash_attn_type = flashAttention ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_AUTO
        cp.n_threads = Int32(ProcessInfo.processInfo.activeProcessorCount)
        cp.n_threads_batch = cp.n_threads

        guard let ctx = llama_init_from_model(model, cp) else {
            throw EngineError(message: "Failed to create context (n_ctx=\(cp.n_ctx), n_seq=\(nSeq))")
        }
        defer { llama_free(ctx) }
        let mem = llama_get_memory(ctx)

        let smpl = llama_sampler_chain_init(llama_sampler_chain_default_params())
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy())
        defer { llama_sampler_free(smpl) }

        var batch = llama_batch_init(nBatch, 0, Int32(nSeq))
        defer { llama_batch_free(batch) }

        func batchClear() { batch.n_tokens = 0 }
        func batchAdd(_ token: llama_token, pos: llama_pos, seq: llama_seq_id, logits: Bool) {
            let i = Int(batch.n_tokens)
            batch.token[i] = token
            batch.pos[i] = pos
            batch.n_seq_id[i] = 1
            batch.seq_id[i]![0] = seq
            batch.logits[i] = logits ? 1 : 0
            batch.n_tokens += 1
        }

        final class Slot {
            var active = false
            var reqIndex = -1
            var seqId: llama_seq_id = 0
            var pos: llama_pos = 0
            var lastToken: llama_token = 0
            var generated = 0
            var promptCount = 0
            var prefillSeconds = 0.0
            var firstTokenAt = 0.0
            var lastTokenAt = 0.0
            var outputPreview = ""
        }

        let slots = (0..<nSeq).map { i -> Slot in
            let s = Slot()
            s.seqId = llama_seq_id(i)
            return s
        }

        var results = [BenchRequestResult]()
        var nextReq = 0
        let clock = { CFAbsoluteTimeGetCurrent() }
        let t0 = clock()
        let availMemBefore = Int64(os_proc_available_memory())

        while true {
            if isCancelled() { throw EngineError(message: "Cancelled") }

            // Admit new requests into idle slots (sequential prefill).
            for slot in slots where !slot.active && nextReq < prompts.count {
                let toks = prompts[nextReq]
                slot.active = true
                slot.reqIndex = nextReq
                slot.promptCount = toks.count
                slot.generated = 0
                slot.outputPreview = ""
                nextReq += 1
                llama_memory_seq_rm(mem, slot.seqId, -1, -1)

                let prefillStart = clock()
                var i = 0
                while i < toks.count {
                    if isCancelled() { throw EngineError(message: "Cancelled") }
                    batchClear()
                    let n = min(Int(nBatch), toks.count - i)
                    for j in 0..<n {
                        batchAdd(toks[i + j], pos: llama_pos(i + j), seq: slot.seqId, logits: i + j == toks.count - 1)
                    }
                    let rc = llama_decode(ctx, batch)
                    guard rc == 0 else { throw EngineError(message: "llama_decode failed (\(rc)) during prefill") }
                    i += n
                }
                // First token comes from the prefill logits.
                let tok = llama_sampler_sample(smpl, ctx, batch.n_tokens - 1)
                let now = clock()
                slot.prefillSeconds = now - prefillStart
                slot.firstTokenAt = now
                slot.lastTokenAt = now
                slot.lastToken = tok
                slot.generated = 1
                slot.pos = llama_pos(toks.count)
                slot.outputPreview = piece(for: tok)
                progress("req \(slot.reqIndex + 1)/\(prompts.count) prefilled \(toks.count) tok in \(String(format: "%.2f", slot.prefillSeconds))s")
            }

            let active = slots.filter { $0.active }
            if active.isEmpty { break }

            // One decode step: one token per active sequence.
            batchClear()
            for slot in active {
                batchAdd(slot.lastToken, pos: slot.pos, seq: slot.seqId, logits: true)
            }
            let rc = llama_decode(ctx, batch)
            guard rc == 0 else { throw EngineError(message: "llama_decode failed (\(rc)) during decode") }
            let now = clock()
            for (k, slot) in active.enumerated() {
                let tok = llama_sampler_sample(smpl, ctx, Int32(k))
                slot.lastToken = tok
                slot.pos += 1
                slot.generated += 1
                slot.lastTokenAt = now
                if slot.outputPreview.utf8.count < 200 {
                    slot.outputPreview += piece(for: tok)
                }
                if slot.generated >= genTokens {
                    llama_memory_seq_rm(mem, slot.seqId, -1, -1)
                    slot.active = false
                    results.append(BenchRequestResult(
                        index: slot.reqIndex,
                        promptTokens: slot.promptCount,
                        genTokens: slot.generated,
                        prefillSeconds: slot.prefillSeconds,
                        ttftSeconds: slot.firstTokenAt - t0,
                        decodeSeconds: slot.lastTokenAt - slot.firstTokenAt,
                        outputPreview: String(slot.outputPreview.prefix(200))
                    ))
                    progress("req \(slot.reqIndex + 1)/\(prompts.count) done (\(results.count)/\(prompts.count) complete)")
                }
            }
        }

        let wall = clock() - t0
        let availMemAfter = Int64(os_proc_available_memory())
        results.sort { $0.index < $1.index }

        return BenchRunResult(
            timestamp: Date(),
            modelName: modelName,
            concurrency: concurrency,
            requestCount: prompts.count,
            genTokensPerRequest: genTokens,
            wallSeconds: wall,
            requests: results,
            nCtx: Int(cp.n_ctx),
            nThreads: Int(cp.n_threads),
            availableMemBeforeMB: Double(availMemBefore) / 1_048_576,
            availableMemAfterMB: Double(availMemAfter) / 1_048_576,
            thermalState: ProcessInfo.processInfo.thermalState.label,
            thermalStateStart: thermalStart,
            kvUnified: kvUnified,
            flashAttention: flashAttention,
            nUbatch: nUbatch
        )
    }
}

extension ProcessInfo.ThermalState {
    var label: String {
        switch self {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}
