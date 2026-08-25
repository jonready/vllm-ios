import Foundation

struct BenchRequestResult: Codable, Identifiable {
    var id: Int { index }
    let index: Int
    let promptTokens: Int
    let genTokens: Int
    let prefillSeconds: Double
    let ttftSeconds: Double     // from run start (all requests submitted at t=0)
    let decodeSeconds: Double   // first token -> last token
    let outputPreview: String

    var prefillTPS: Double { prefillSeconds > 0 ? Double(promptTokens) / prefillSeconds : 0 }
    var decodeTPS: Double { decodeSeconds > 0 ? Double(genTokens - 1) / decodeSeconds : 0 }
}

struct BenchRunResult: Codable, Identifiable {
    var id: Date { timestamp }
    let timestamp: Date
    let modelName: String
    let concurrency: Int
    let requestCount: Int
    let genTokensPerRequest: Int
    let wallSeconds: Double
    let requests: [BenchRequestResult]
    let nCtx: Int
    let nThreads: Int
    let availableMemBeforeMB: Double
    let availableMemAfterMB: Double
    let thermalState: String
    // Optional so result files from older app builds still decode.
    let thermalStateStart: String?
    let kvUnified: Bool?
    let flashAttention: Bool?
    var nUbatch: Int? = nil
    var stageLabel: String? = nil
    var engine: String? = nil   // nil/"llama.cpp" vs "mlx"
    // Burst mode: seconds idling after this run until thermal state returned
    // to the pre-run state (nil = not measured / not burst mode).
    var recoverySeconds: Double? = nil
    var burstIndex: Int? = nil

    var configLabel: String {
        var parts: [String] = []
        if kvUnified == true { parts.append("kvU") }
        if flashAttention == true { parts.append("FA") }
        return parts.isEmpty ? "" : parts.joined(separator: "+")
    }

    // MARK: - Aggregates

    var totalPromptTokens: Int { requests.reduce(0) { $0 + $1.promptTokens } }
    var totalGenTokens: Int { requests.reduce(0) { $0 + $1.genTokens } }
    var totalPrefillSeconds: Double { requests.reduce(0) { $0 + $1.prefillSeconds } }

    /// Prompt tokens per second, aggregated (prefills are sequential in the engine).
    var aggregatePrefillTPS: Double {
        totalPrefillSeconds > 0 ? Double(totalPromptTokens) / totalPrefillSeconds : 0
    }

    /// Generated tokens per second over the non-prefill portion of the run.
    var aggregateDecodeTPS: Double {
        let decodeWall = wallSeconds - totalPrefillSeconds
        return decodeWall > 0 ? Double(totalGenTokens) / decodeWall : 0
    }

    /// All tokens (prompt + generated) per wall-clock second.
    var overallTPS: Double {
        wallSeconds > 0 ? Double(totalPromptTokens + totalGenTokens) / wallSeconds : 0
    }

    var meanPerStreamDecodeTPS: Double {
        guard !requests.isEmpty else { return 0 }
        return requests.map(\.decodeTPS).reduce(0, +) / Double(requests.count)
    }

    var ttftP50: Double { percentile(requests.map(\.ttftSeconds), 0.50) }
    var ttftP95: Double { percentile(requests.map(\.ttftSeconds), 0.95) }
    var ttftMin: Double { requests.map(\.ttftSeconds).min() ?? 0 }
    var ttftMax: Double { requests.map(\.ttftSeconds).max() ?? 0 }

    private func percentile(_ values: [Double], _ p: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let idx = min(sorted.count - 1, Int((Double(sorted.count) * p).rounded(.up)) - 1)
        return sorted[max(0, idx)]
    }
}

struct BenchSession: Codable {
    let device: String
    let os: String
    let llamaBuild: String
    var runs: [BenchRunResult]
}

enum DeviceInfo {
    static var machine: String {
        var sys = utsname()
        uname(&sys)
        return withUnsafePointer(to: &sys.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 256) { String(cString: $0) }
        }
    }
}
