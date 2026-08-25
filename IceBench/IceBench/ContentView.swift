import SwiftUI

struct ContentView: View {
    @EnvironmentObject var bench: BenchController
    @EnvironmentObject var models: ModelManager

    var body: some View {
        NavigationStack {
            Form {
                modelSection
                sweepSection
                configSection
                runSection
                logSection
                resultsSection
            }
            .navigationTitle("IceBench")
        }
    }

    // MARK: - Model

    private var modelSection: some View {
        Section("Model library") {
            // Active models, one line per engine.
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    engineTag("llama.cpp", color: .blue)
                    Text(bench.loadedModelName ?? "none loaded")
                        .font(.caption)
                        .foregroundStyle(bench.loadedModelName == nil ? .secondary : .primary)
                }
                if !bench.modelInfo.isEmpty {
                    Text(bench.modelInfo).font(.caption2).foregroundStyle(.secondary)
                }
                HStack(spacing: 6) {
                    engineTag("MLX", color: .purple)
                    Text(bench.mlxLoadedId?.replacingOccurrences(of: "mlx-community/", with: "") ?? "none loaded")
                        .font(.caption)
                        .foregroundStyle(bench.mlxLoadedId == nil ? .secondary : .primary)
                }
            }

            // GGUF quants (llama.cpp engine)
            ForEach(ModelManager.presets, id: \.url) { preset in
                let filename = URL(string: preset.url)?.lastPathComponent ?? ""
                let local = models.downloadedModels.first { $0.lastPathComponent == filename }
                modelRow(
                    name: preset.name,
                    engine: "llama.cpp", color: .blue,
                    sizeMB: local.flatMap { try? $0.resourceValues(forKeys: [.fileSizeKey]).fileSize }
                        .map { Double($0) / 1_048_576 },
                    isLoaded: bench.loadedModelName == filename,
                    actionLabel: local == nil ? "Get" : "Load"
                ) {
                    if let local { bench.loadModel(at: local) }
                    else { models.download(urlString: preset.url) }
                }
            }
            // Any other local .gguf files (e.g. dropped in via Files)
            ForEach(models.downloadedModels.filter { url in
                !ModelManager.presets.contains { URL(string: $0.url)?.lastPathComponent == url.lastPathComponent }
            }, id: \.self) { url in
                modelRow(
                    name: url.lastPathComponent,
                    engine: "llama.cpp", color: .blue,
                    sizeMB: (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize)
                        .map { Double($0) / 1_048_576 },
                    isLoaded: bench.loadedModelName == url.lastPathComponent,
                    actionLabel: "Load"
                ) { bench.loadModel(at: url) }
                .swipeActions {
                    Button("Delete", role: .destructive) { models.delete(url) }
                }
            }

            // MLX quants (downloaded from HF on first load)
            ForEach(MLXEngine.presets, id: \.id) { preset in
                modelRow(
                    name: preset.label,
                    engine: "MLX", color: .purple,
                    sizeMB: nil,
                    isLoaded: bench.mlxLoadedId == preset.id,
                    actionLabel: "Load"
                ) { bench.loadMLXModel(id: preset.id) }
            }

            if let progress = models.downloadProgress {
                VStack(alignment: .leading) {
                    ProgressView(value: progress)
                    HStack {
                        Text(models.downloadStatus).font(.caption).foregroundStyle(.secondary)
                        Spacer()
                        Button("Cancel") { models.cancelDownload() }.font(.caption)
                    }
                }
            } else {
                HStack {
                    TextField("Custom GGUF URL", text: $models.customURL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .font(.caption)
                    Button("Get") { models.download(urlString: models.customURL) }
                        .disabled(models.customURL.isEmpty)
                }
                if !models.downloadStatus.isEmpty {
                    Text(models.downloadStatus).font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }

    private func engineTag(_ name: String, color: Color) -> some View {
        Text(name)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 5).padding(.vertical, 1)
            .background(color.opacity(0.15))
            .foregroundStyle(color)
            .clipShape(Capsule())
    }

    private func modelRow(
        name: String, engine: String, color: Color, sizeMB: Double?,
        isLoaded: Bool, actionLabel: String, action: @escaping () -> Void
    ) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 1) {
                Text(name).font(.caption)
                HStack(spacing: 6) {
                    engineTag(engine, color: color)
                    if let sizeMB {
                        Text(String(format: "%.0f MB", sizeMB))
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }
            }
            Spacer()
            if isLoaded {
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
            } else {
                Button(actionLabel, action: action)
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(bench.isBusy)
            }
        }
    }

    // MARK: - Automated sweeps

    private var sweepSection: some View {
        Section("Automated sweeps (one tap, hands off)") {
            VStack(alignment: .leading, spacing: 4) {
                Button {
                    bench.startScalingSweep()
                } label: {
                    Label("Scaling sweep — phone ON ice pack", systemImage: "snowflake")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(bench.loadedModelName == nil)
                Text("base config, c1/2/4/8 with mandatory 90s gaps (die cools even when thermal state reads nominal). Medium prompts, gen 128, 8 req. ~12 min.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            VStack(alignment: .leading, spacing: 4) {
                Button {
                    bench.startMLXBench()
                } label: {
                    Label("MLX arm — Qwen3.5-0.8B", systemImage: "m.square")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.purple)
                .disabled(bench.mlxLoadedId == nil)
                Text("Runs the loaded MLX model: \(bench.mlxLoadedId?.replacingOccurrences(of: "mlx-community/Qwen3.5-0.8B-", with: "") ?? "load one in the library above"). Single stream, 8 sequential requests, medium prompts, gen 128, greedy. Compare against llama.cpp c1.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            VStack(alignment: .leading, spacing: 4) {
                Button {
                    bench.startMLXBatchedBench()
                } label: {
                    Label("MLX BATCHED — B=8 lockstep", systemImage: "square.stack.3d.up.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.indigo)
                .disabled(bench.mlxLoadedId == nil)
                Text("DIY batched decode on the loaded MLX model: 8 equal-length prompts prefilled as one [8, L] batch, then lockstep [8, 1] greedy decode. The experiment: does aggregate t/s scale past single-stream 103?")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            VStack(alignment: .leading, spacing: 4) {
                Button {
                    bench.startMLXScalingSweep()
                } label: {
                    Label("MLX scaling sweep — b1/2/4/8 (ice pack)", systemImage: "chart.line.uptrend.xyaxis")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.mint)
                .disabled(bench.mlxLoadedId == nil)
                Text("Engine burst of 8 requests at batch caps 1/2/4/8, 90s gaps + thermal gate — mirrors the llama.cpp scaling sweep level for level.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            VStack(alignment: .leading, spacing: 4) {
                Button {
                    bench.startVLLMBench()
                } label: {
                    Label("vllm-swift engine — burst + stagger", systemImage: "arrow.triangle.branch")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.teal)
                .disabled(bench.mlxLoadedId == nil)
                Text("Continuous-batching engine (join/exit at token boundaries): 16 requests as an all-at-once burst, then again arriving one per 0.5s. Batch cap = highest selected concurrency level above. Cools to fair between scenarios.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            VStack(alignment: .leading, spacing: 4) {
                Button {
                    bench.startDutySweep()
                } label: {
                    Label("Duty-cycle sweep — phone OFF ice pack", systemImage: "flame")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.orange)
                .disabled(bench.loadedModelName == nil)
                Text("c8 bursts, gen 64: gap 0s ×3, 30s ×4, 90s ×4. Phone flat on a table, screen dim. ~15 min.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
        .disabled(bench.isBusy)
    }

    // MARK: - Config

    private var configSection: some View {
        Section("Workload — blog research (prefill heavy)") {
            Picker("Prompt size", selection: $bench.preset) {
                ForEach(PromptPreset.allCases) { p in
                    Text(p.rawValue).tag(p)
                }
            }
            Stepper("Generate \(bench.genTokens) tokens", value: $bench.genTokens, in: 16...512, step: 16)
            Stepper("\(bench.requestCount) requests per level", value: $bench.requestCount, in: 1...32)
            VStack(alignment: .leading, spacing: 6) {
                Text("Concurrency levels")
                HStack {
                    ForEach([1, 2, 4, 8], id: \.self) { level in
                        let on = bench.concurrencyLevels.contains(level)
                        Button("\(level)") {
                            if on { bench.concurrencyLevels.remove(level) }
                            else { bench.concurrencyLevels.insert(level) }
                        }
                        .buttonStyle(.bordered)
                        .tint(on ? .accentColor : .gray)
                    }
                }
            }
            Toggle("Thermal gate (cool to fair before each level)", isOn: $bench.thermalGate)
            Toggle("Unified KV cache (kv_unified)", isOn: $bench.kvUnified)
            Toggle("Force flash attention", isOn: $bench.flashAttention)
            Picker("n_ubatch", selection: $bench.nUbatch) {
                ForEach([512, 1024, 2048], id: \.self) { Text("\($0)").tag($0) }
            }
            Toggle("Prefix cache (engine runs: share the system prompt KV)", isOn: $bench.usePrefixCache)
            Toggle("Burst mode (all requests in one batch, time recovery)", isOn: $bench.burstMode)
            if bench.burstMode {
                Stepper("\(bench.burstCount) bursts", value: $bench.burstCount, in: 1...10)
                Stepper("\(bench.burstGapSeconds)s min gap between bursts", value: $bench.burstGapSeconds, in: 0...180, step: 15)
            }
        }
        .disabled(bench.isBusy)
    }

    // MARK: - Run

    private var runSection: some View {
        Section {
            switch bench.state {
            case .idle:
                Button {
                    bench.runBenchmarks()
                } label: {
                    Label("Run benchmark", systemImage: "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(bench.loadedModelName == nil || bench.concurrencyLevels.isEmpty)
            case .loadingModel:
                HStack { ProgressView(); Text("Loading model…") }
            case .running(let status):
                VStack(alignment: .leading, spacing: 6) {
                    HStack { ProgressView(); Text(status).font(.caption) }
                    Button("Stop", role: .destructive) { bench.cancel() }
                }
            }
        }
    }

    // MARK: - Log

    private var logSection: some View {
        Section("Log") {
            if bench.log.isEmpty {
                Text("—").foregroundStyle(.secondary)
            } else {
                ForEach(Array(bench.log.suffix(12).enumerated()), id: \.offset) { _, line in
                    Text(line)
                        .font(.system(size: 11, design: .monospaced))
                        .lineLimit(4)
                }
            }
        }
    }

    // MARK: - Results

    private var resultsSection: some View {
        Section {
            ForEach(bench.runs) { run in
                NavigationLink {
                    RunDetailView(run: run)
                } label: {
                    RunRowView(run: run)
                }
            }
        } header: {
            HStack {
                Text("Results")
                Spacer()
                if let url = bench.exportURL {
                    ShareLink(item: url) { Image(systemName: "square.and.arrow.up") }
                }
                if !bench.runs.isEmpty {
                    Button { bench.clearResults() } label: { Image(systemName: "trash") }
                }
            }
        }
    }
}

struct RunRowView: View {
    let run: BenchRunResult

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text((run.stageLabel.map { "\($0) · " } ?? "") +
                     (run.burstIndex.map { "burst #\($0) · " } ?? "") +
                     "c\(run.concurrency) × \(run.requestCount) req" +
                     (run.configLabel.isEmpty ? "" : " · \(run.configLabel)"))
                    .font(.subheadline.weight(.semibold))
                Text(run.thermalStateStart ?? "")
                    .font(.caption2).foregroundStyle(.orange)
                Spacer()
                Text(run.timestamp, format: .dateTime.hour().minute().second())
                    .font(.caption2).foregroundStyle(.secondary)
            }
            Text(run.engine == "vllm-swift"
                 ? String(format: "TTFT p50 %.2fs · decode %.1f t/s/str · gen/wall %.1f t/s · overall %.1f t/s",
                          run.ttftP50, run.meanPerStreamDecodeTPS,
                          run.aggregateDecodeTPS, run.overallTPS)
                 : String(format: "TTFT p50 %.2fs · decode %.1f t/s/str · agg %.1f t/s · overall %.1f t/s",
                          run.ttftP50, run.meanPerStreamDecodeTPS,
                          run.aggregateDecodeTPS, run.overallTPS))
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}
