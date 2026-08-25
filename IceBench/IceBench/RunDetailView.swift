import SwiftUI

struct RunDetailView: View {
    let run: BenchRunResult

    var body: some View {
        List {
            Section("Run") {
                row("Model", run.modelName)
                row("Concurrency", "\(run.concurrency)")
                row("Requests", "\(run.requestCount)")
                row("Gen tokens/req", "\(run.genTokensPerRequest)")
                row("Wall time", String(format: "%.2f s", run.wallSeconds))
                row("n_ctx / threads", "\(run.nCtx) / \(run.nThreads)")
                row("Thermal start → end", "\(run.thermalStateStart ?? "?") → \(run.thermalState)")
                if let recovery = run.recoverySeconds {
                    row("Thermal recovery", String(format: "%.0f s (duty %.0f%%)",
                        recovery, run.wallSeconds / (run.wallSeconds + recovery) * 100))
                }
                if let ub = run.nUbatch { row("n_ubatch", "\(ub)") }
                row("KV unified / flash attn", "\(run.kvUnified == true ? "yes" : "no") / \(run.flashAttention == true ? "yes" : "no")")
                row("Free app mem", String(format: "%.0f → %.0f MB",
                                           run.availableMemBeforeMB, run.availableMemAfterMB))
            }

            Section("Throughput") {
                row("Prompt tokens", "\(run.totalPromptTokens)")
                row("Generated tokens", "\(run.totalGenTokens)")
                row("Prefill (aggregate)", String(format: "%.0f tok/s", run.aggregatePrefillTPS))
                row("Decode per stream (mean)", String(format: "%.1f tok/s", run.meanPerStreamDecodeTPS))
                row("Decode aggregate", String(format: "%.1f tok/s", run.aggregateDecodeTPS))
                row("Overall (all tokens / wall)", String(format: "%.1f tok/s", run.overallTPS))
            }

            Section("TTFT (from run start)") {
                row("min", String(format: "%.2f s", run.ttftMin))
                row("p50", String(format: "%.2f s", run.ttftP50))
                row("p95", String(format: "%.2f s", run.ttftP95))
                row("max", String(format: "%.2f s", run.ttftMax))
            }

            Section("Requests") {
                ForEach(run.requests) { req in
                    VStack(alignment: .leading, spacing: 3) {
                        Text(String(format: "#%d · %d ptok · prefill %.2fs (%.0f t/s) · TTFT %.2fs · decode %.1f t/s",
                                    req.index + 1, req.promptTokens, req.prefillSeconds,
                                    req.prefillTPS, req.ttftSeconds, req.decodeTPS))
                            .font(.system(size: 11, design: .monospaced))
                        Text(req.outputPreview)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(3)
                    }
                }
            }
        }
        .navigationTitle("c\(run.concurrency) run")
        .navigationBarTitleDisplayMode(.inline)
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
            Spacer()
            Text(value).foregroundStyle(.secondary)
        }
        .font(.subheadline)
    }
}
