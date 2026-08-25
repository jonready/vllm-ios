import Foundation
import MLX
import MLXLMCommon

// vllm-swift: a continuous-batching inference engine on mlx-swift-lm,
// architecturally modeled on vllm-mlx (arXiv:2601.19139) but sharing no code.
//
// Design: UNIFORM-OFFSET continuous batching. Every sequence in the running
// batch always has the same KV offset, so the model's stock causal-mask path
// works unchanged (no ragged masks, no model fork):
//
//   - ADMIT at a token boundary: the arrival's prompt is left-padded with
//     filler tokens to exactly the group's current offset, prefilled solo
//     (chunked, cache-state-only eval), then spliced into the batch by cache
//     surgery (concat along the batch axis). An arrival whose prompt is longer
//     than the current offset waits in queue — the offset grows by 1 every
//     step, so it becomes admissible shortly.
//   - EXIT at a token boundary: finished sequences are sliced out of every
//     cache (index-select along the batch axis) and the batch shrinks.
//
// Cache surgery uses public API only: ArraysCache (Mamba/GDN recurrent state)
// has native extend/filter; KVCacheSimple goes through the `state` accessor.

public struct EngineRequest {
    public let id: Int
    public let promptTokens: [Int32]
    public let maxTokens: Int
    public let arrivalTime: Double  // CFAbsoluteTime; engine admits no earlier

    public init(id: Int, promptTokens: [Int32], maxTokens: Int, arrivalTime: Double) {
        self.id = id
        self.promptTokens = promptTokens
        self.maxTokens = maxTokens
        self.arrivalTime = arrivalTime
    }
}

/// A frozen KV/recurrent-cache snapshot of a shared prompt prefix (batch 1).
/// Build once with ``VLLMEngine/cachePrefix(_:)``; pass to ``VLLMEngine/run(requests:prefix:)``
/// and every admitted request pays prefill only for its suffix.
public final class PrefixCache {
    public let tokens: [Int32]
    let caches: [KVCache]

    init(tokens: [Int32], caches: [KVCache]) {
        self.tokens = tokens
        self.caches = caches
    }

    public var tokenCount: Int { tokens.count }

    /// Longest common token prefix across prompts, capped so every prompt
    /// keeps at least one suffix token to prefill.
    public static func longestCommonPrefix(of prompts: [[Int32]]) -> [Int32] {
        guard let first = prompts.first, !first.isEmpty else { return [] }
        var n = first.count
        for p in prompts.dropFirst() {
            var i = 0
            let bound = min(n, p.count)
            while i < bound && p[i] == first[i] { i += 1 }
            n = i
            if n == 0 { return [] }
        }
        let minLen = prompts.map(\.count).min() ?? 0
        n = min(n, minLen - 1)
        return n > 0 ? Array(first[..<n]) : []
    }
}

public struct EngineResult {
    public let id: Int
    public let promptLength: Int
    public let paddedLength: Int
    /// Tokens served from the shared prefix cache instead of being prefilled.
    public let prefixTokens: Int
    public let tokens: [Int32]
    public let arrivalTime: Double
    public let firstTokenTime: Double
    public let completionTime: Double

    public var ttft: Double { firstTokenTime - arrivalTime }
    public var decodeSeconds: Double { completionTime - firstTokenTime }
    public var decodeTPS: Double {
        tokens.count > 1 && decodeSeconds > 0 ? Double(tokens.count - 1) / decodeSeconds : 0
    }
}

public struct StepStat {
    public let time: Double
    public let batchSize: Int
    public let stepSeconds: Double
}

public struct EngineRunReport {
    public let results: [EngineResult]
    public let stepStats: [StepStat]
    public let events: [String]
    public let wallSeconds: Double
}

public final class VLLMEngine {
    private let model: any LanguageModel
    private let padToken: Int32
    private let eosTokens: Set<Int32>
    public var maxBatch: Int
    public var prefillChunk = 512
    /// Number of greedy decode steps chained into one lazy graph before a
    /// single eval + readback. The argmax→input feedback stays on-GPU, so the
    /// CPU only syncs once per chunk. Exits/admissions happen at chunk
    /// granularity; the chunk is clamped so no stream overshoots maxTokens.
    public var decodeChunk = 8
    public var log: (String) -> Void = { _ in }

    public init(
        model: any LanguageModel, padToken: Int32, eosTokens: Set<Int32> = [],
        maxBatch: Int = 8
    ) {
        self.model = model
        self.padToken = padToken
        self.eosTokens = eosTokens
        self.maxBatch = maxBatch
    }

    // MARK: - Cache surgery

    private func joinCaches(group: [KVCache], arrival: [KVCache]) {
        for (g, a) in zip(group, arrival) {
            if let gm = g as? ArraysCache, let am = a as? ArraysCache {
                gm.extend(other: am)
            } else {
                let gs = g.state
                let asv = a.state
                precondition(gs.count == asv.count, "cache state arity mismatch")
                var gm = g  // KVCache conformers are classes; shadow to satisfy the setter
                gm.state = zip(gs, asv).map { concatenated([$0, $1], axis: 0) }
            }
        }
    }

    private func filterCaches(_ caches: [KVCache], keep: [Int]) {
        let idx = MLXArray(keep.map(Int32.init))
        for c in caches {
            if let m = c as? ArraysCache {
                m.filter(batchIndices: idx)
            } else {
                var cm = c
                cm.state = c.state.map { $0[idx] }
            }
        }
    }

    // MARK: - Prefix caching

    /// Prefill `tokens` once (batch 1, cache-state-only evaluation) and freeze
    /// the result. The returned object is immutable; runs tile deep copies of
    /// it, so one PrefixCache serves any number of runs and batch sizes.
    public func cachePrefix(_ tokens: [Int32]) throws -> PrefixCache {
        precondition(!tokens.isEmpty, "prefix must be non-empty")
        let caches = try model.newCache(parameters: nil)
        let inputs = MLXArray(tokens).reshaped([1, tokens.count])
        var pos = 0
        while pos < tokens.count {
            let n = min(prefillChunk, tokens.count - pos)
            _ = model(inputs[0..., pos ..< (pos + n)], cache: caches)
            eval(caches.flatMap { $0.innerState() })
            pos += n
        }
        return PrefixCache(tokens: tokens, caches: caches)
    }

    /// Batch-k caches primed with the prefix: deep-copy the frozen batch-1
    /// snapshot and concatenate copies along the batch axis via the same
    /// surgery used for admission.
    private func tiledCaches(_ prefix: PrefixCache, count: Int) -> [KVCache] {
        let base = prefix.caches.map { $0.copy() }
        guard count > 1 else { return base }
        for _ in 1..<count {
            joinCaches(group: base, arrival: prefix.caches.map { $0.copy() })
        }
        return base
    }

    // MARK: - Prefill (solo batch of arrivals, all padded to targetLen)

    /// Prefill `padded` token rows into `caches` (fresh, or prefix-primed with
    /// a nonzero offset); returns the first sampled token per row.
    private func prefill(_ padded: [[Int32]], into caches: [KVCache]) throws -> MLXArray {
        let B = padded.count
        let L = padded[0].count
        let inputs = MLXArray(padded.flatMap { $0 }).reshaped([B, L])

        var pos = 0
        let lastIndex = L - 1
        while pos < lastIndex {
            let n = min(prefillChunk, lastIndex - pos)
            _ = model(inputs[0..., pos ..< (pos + n)], cache: caches)
            eval(caches.flatMap { $0.innerState() })
            pos += n
        }
        let logits = model(inputs[0..., lastIndex ..< L], cache: caches)
        let first = argMax(logits[0..., logits.dim(1) - 1, 0...], axis: -1).asType(.int32)
        eval(first)
        return first
    }

    // MARK: - Run

    private final class Slot {
        let request: EngineRequest
        var lastToken: Int32
        var tokens: [Int32]
        let paddedLength: Int
        let firstTokenTime: Double

        init(request: EngineRequest, firstToken: Int32, paddedLength: Int, at time: Double) {
            self.request = request
            self.lastToken = firstToken
            self.tokens = [firstToken]
            self.paddedLength = paddedLength
            self.firstTokenTime = time
        }

        var isDone: Bool { tokens.count >= request.maxTokens }
    }

    /// Runs the engine until every request has completed. Single-threaded;
    /// arrivals are honored by their `arrivalTime` (virtual open-loop load).
    /// With a `prefix`, every request's prompt must start with the prefix
    /// tokens; admission tiles the frozen prefix cache across the batch and
    /// prefills only each request's suffix.
    public func run(requests: [EngineRequest], prefix: PrefixCache? = nil) throws -> EngineRunReport {
        let prefixLen = prefix?.tokenCount ?? 0
        if let prefix {
            for req in requests {
                precondition(
                    req.promptTokens.count > prefixLen
                        && Array(req.promptTokens.prefix(prefixLen)) == prefix.tokens,
                    "request \(req.id) does not start with the shared prefix")
            }
        }
        var queue = requests.sorted { $0.arrivalTime < $1.arrivalTime }
        var slots: [Slot] = []
        var groupCaches: [KVCache]? = nil
        var offset = 0
        var results: [EngineResult] = []
        var stepStats: [StepStat] = []
        var events: [String] = []
        let t0 = CFAbsoluteTimeGetCurrent()

        func note(_ s: String) {
            let stamp = String(format: "%7.2fs", CFAbsoluteTimeGetCurrent() - t0)
            events.append("\(stamp) \(s)")
            log(s)
        }

        while !queue.isEmpty || !slots.isEmpty {
            let now = CFAbsoluteTimeGetCurrent()

            // ---- Admission at this token boundary ----
            // Eligible: arrived, room in batch, and (if a group is running)
            // prompt fits inside the current uniform offset.
            var admitted: [EngineRequest] = []
            while slots.count + admitted.count < maxBatch,
                let next = queue.first,
                next.arrivalTime <= now,
                slots.isEmpty || next.promptTokens.count <= offset
            {
                admitted.append(next)
                queue.removeFirst()
            }

            if !admitted.isEmpty {
                // Pad to the group offset (or, for a fresh group, the longest
                // arrival). Padding is spliced ahead of the last 20 tokens so
                // the chat-template tail stays intact.
                let targetLen = slots.isEmpty
                    ? admitted.map { $0.promptTokens.count }.max()!
                    : offset
                // With a prefix, pad and prefill only the suffix; the prefix
                // rows arrive pre-filled via the tiled cache snapshot.
                let suffixTarget = targetLen - prefixLen
                let padded = admitted.map { req -> [Int32] in
                    let toks = Array(req.promptTokens.dropFirst(prefixLen))
                    let need = suffixTarget - toks.count
                    guard need > 0 else { return toks }
                    let cut = max(0, toks.count - 20)
                    return Array(toks[..<cut])
                        + Array(repeating: padToken, count: need)
                        + Array(toks[cut...])
                }
                let tPrefill = CFAbsoluteTimeGetCurrent()
                let newCaches: [KVCache]
                if let prefix {
                    newCaches = tiledCaches(prefix, count: admitted.count)
                } else {
                    newCaches = try model.newCache(parameters: nil)
                }
                let firstTokens = try prefill(padded, into: newCaches)
                let firstArr = firstTokens.asArray(Int32.self)
                let tokenTime = CFAbsoluteTimeGetCurrent()

                for (k, req) in admitted.enumerated() {
                    slots.append(Slot(
                        request: req, firstToken: firstArr[k],
                        paddedLength: targetLen, at: tokenTime))
                }
                if let g = groupCaches {
                    joinCaches(group: g, arrival: newCaches)
                } else {
                    groupCaches = newCaches
                    offset = targetLen
                }
                note("admit \(admitted.map(\.id)) @offset \(targetLen) " +
                     "(prefill \(String(format: "%.2f", tokenTime - tPrefill))s, batch → \(slots.count))")
            }

            guard let caches = groupCaches, !slots.isEmpty else {
                if queue.isEmpty { break }
                // Nothing running and the next arrival is in the future: wait.
                Thread.sleep(forTimeInterval: max(0, queue[0].arrivalTime - now))
                continue
            }

            // ---- One decode chunk: K lockstep steps, single eval ----
            // Clamp so no stream generates past its maxTokens, and admit
            // pending arrivals reasonably promptly.
            let remaining = slots.map { $0.request.maxTokens - $0.tokens.count }.min() ?? 1
            let k = max(1, min(decodeChunk, remaining))
            let stepStart = CFAbsoluteTimeGetCurrent()
            var cur = MLXArray(slots.map(\.lastToken)).reshaped([slots.count, 1])
            var chunkTokens: [MLXArray] = []
            for _ in 0..<k {
                let logits = model(cur, cache: caches)
                let next = argMax(logits[0..., logits.dim(1) - 1, 0...], axis: -1).asType(.int32)
                chunkTokens.append(next)
                cur = next.reshaped([slots.count, 1])
            }
            eval(chunkTokens)
            let stepEnd = CFAbsoluteTimeGetCurrent()
            offset += k
            stepStats.append(StepStat(
                time: stepEnd - t0, batchSize: slots.count,
                stepSeconds: (stepEnd - stepStart) / Double(k)))

            var finished: [Int] = []
            let stepArrs = chunkTokens.map { $0.asArray(Int32.self) }
            for (i, slot) in slots.enumerated() {
                for arr in stepArrs {
                    // A stream that hit EOS mid-chunk stops accumulating; the
                    // extra positions in its cache are discarded at exit.
                    if slot.isDone || (slot.tokens.last.map { eosTokens.contains($0) } ?? false) {
                        break
                    }
                    slot.lastToken = arr[i]
                    slot.tokens.append(arr[i])
                }
                if slot.isDone || eosTokens.contains(slot.lastToken) {
                    finished.append(i)
                }
            }

            // ---- Exit at this token boundary ----
            if !finished.isEmpty {
                let keep = (0..<slots.count).filter { !finished.contains($0) }
                for i in finished {
                    let slot = slots[i]
                    results.append(EngineResult(
                        id: slot.request.id,
                        promptLength: slot.request.promptTokens.count,
                        paddedLength: slot.paddedLength,
                        prefixTokens: prefixLen,
                        tokens: slot.tokens,
                        arrivalTime: slot.request.arrivalTime,
                        firstTokenTime: slot.firstTokenTime,
                        completionTime: stepEnd))
                }
                note("exit \(finished.map { slots[$0].request.id }) (batch → \(keep.count))")
                if keep.isEmpty {
                    groupCaches = nil
                    offset = 0
                    slots = []
                } else {
                    filterCaches(caches, keep: keep)
                    slots = keep.map { slots[$0] }
                }
            }
        }

        return EngineRunReport(
            results: results.sorted { $0.id < $1.id },
            stepStats: stepStats,
            events: events,
            wallSeconds: CFAbsoluteTimeGetCurrent() - t0)
    }
}
