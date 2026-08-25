import SwiftUI

struct ChatView: View {
    @EnvironmentObject var fleet: FleetController
    @State private var draft = ""
    @State private var showSettings = false
    @State private var showChats = false
    @FocusState private var composerFocused: Bool

    var body: some View {
        ZStack {
            heroGradient.ignoresSafeArea()
            VStack(spacing: 0) {
                header
                content
                if fleet.currentChat.turns.isEmpty { suggestionChips }
                composer
            }
        }
        .onAppear { fleet.loadModelIfNeeded() }
        .sheet(isPresented: $showSettings) { SettingsSheet() }
        .sheet(isPresented: $showChats) { ChatsSheet() }
    }

    // Locally-AI-style hero: purple fading into the background color.
    private var heroGradient: some View {
        LinearGradient(
            colors: [
                Color(red: 0.55, green: 0.47, blue: 0.93),
                Color(red: 0.66, green: 0.60, blue: 0.95),
                Color(.systemBackground),
            ],
            startPoint: .top, endPoint: .init(x: 0.5, y: 0.75)
        )
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 12) {
            // Settings + chats pill
            HStack(spacing: 2) {
                Button { showSettings = true } label: {
                    Image(systemName: "gearshape")
                        .font(.system(size: 17, weight: .medium))
                        .frame(width: 40, height: 40)
                }
                Button { showChats = true } label: {
                    Image(systemName: "bubble.left")
                        .font(.system(size: 16, weight: .medium))
                        .frame(width: 40, height: 40)
                }
            }
            .foregroundStyle(.primary)
            .background(.regularMaterial, in: Capsule())

            Spacer()

            // Model selector
            Menu {
                ForEach(FleetController.models) { m in
                    Button {
                        fleet.select(model: m)
                    } label: {
                        if m == fleet.selectedModel {
                            Label("\(m.family) \(m.variant)", systemImage: "checkmark")
                        } else {
                            Text("\(m.family) \(m.variant)")
                        }
                    }
                }
            } label: {
                HStack(spacing: 4) {
                    Text(fleet.selectedModel.family)
                        .font(.headline)
                        .foregroundStyle(.primary)
                    Text(fleet.selectedModel.variant)
                        .font(.headline)
                        .foregroundStyle(.secondary)
                    Image(systemName: "chevron.right")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                }
            }
            .disabled(fleet.isRunning)

            Spacer()

            // New chat
            Button { fleet.newChat() } label: {
                Image(systemName: "square.and.pencil")
                    .font(.system(size: 16, weight: .medium))
                    .frame(width: 44, height: 44)
                    .foregroundStyle(.primary)
                    .background(.regularMaterial, in: Circle())
            }
            .disabled(fleet.isRunning)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 6)
    }

    // MARK: - Content

    @ViewBuilder
    private var content: some View {
        if fleet.currentChat.turns.isEmpty {
            emptyState
        } else {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 20) {
                        ForEach(fleet.currentChat.turns) { turn in
                            TurnView(turn: turn).id(turn.id)
                        }
                        Color.clear.frame(height: 1).id("bottom")
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 12)
                }
                .onChange(of: fleet.currentChat.turns.last?.agents.map(\.text.count).reduce(0, +) ?? 0) {
                    withAnimation(.easeOut(duration: 0.15)) {
                        proxy.scrollTo("bottom", anchor: .bottom)
                    }
                }
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 18) {
            Spacer()
            Text("Meet the Swarm")
                .font(.system(size: 38, weight: .semibold))
            Text("One question. Four specialist agents\nanswering in parallel — entirely on this device.")
                .multilineTextAlignment(.center)
                .font(.title3)
                .foregroundStyle(.secondary)
            statusLine
            Spacer()
            Spacer()
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 24)
    }

    private var statusLine: some View {
        Group {
            switch fleet.modelState {
            case .idle, .loading:
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    if case .loading(let msg) = fleet.modelState {
                        Text(msg)
                    } else {
                        Text("Preparing…")
                    }
                }
            case .ready:
                Label("Ready · on-device", systemImage: "checkmark.circle.fill")
            case .failed(let msg):
                Text(msg).foregroundStyle(.red)
            }
        }
        .font(.footnote)
        .foregroundStyle(.secondary)
        .padding(.top, 8)
    }

    // MARK: - Suggestions

    private var suggestionChips: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                chip("Plan", "a weekend in Kyoto")
                chip("Compare", "electric vs gas cars")
                chip("Explain", "continuous batching")
                chip("Draft", "a cold outreach email")
            }
            .padding(.horizontal, 16)
        }
        .padding(.bottom, 8)
    }

    private func chip(_ title: String, _ subtitle: String) -> some View {
        Button {
            draft = "\(title) \(subtitle)"
            composerFocused = true
        } label: {
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.subheadline.weight(.semibold))
                Text(subtitle).font(.subheadline).foregroundStyle(.secondary)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        }
        .foregroundStyle(.primary)
    }

    // MARK: - Composer

    private var composer: some View {
        HStack(alignment: .bottom, spacing: 10) {
            Button { showSettings = true } label: {
                Image(systemName: "plus")
                    .font(.system(size: 18, weight: .medium))
                    .frame(width: 42, height: 42)
                    .foregroundStyle(.primary)
                    .background(.regularMaterial, in: Circle())
            }
            TextField("Ask anything", text: $draft, axis: .vertical)
                .lineLimit(1...5)
                .focused($composerFocused)
                .padding(.horizontal, 16)
                .padding(.vertical, 11)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
            Button {
                let text = draft
                draft = ""
                fleet.send(text)
            } label: {
                Image(systemName: "arrow.up")
                    .font(.system(size: 17, weight: .semibold))
                    .frame(width: 42, height: 42)
                    .foregroundStyle(canSend ? Color(.systemBackground) : Color(.systemGray2))
                    .background(canSend ? Color.primary : Color(.systemGray5), in: Circle())
            }
            .disabled(!canSend)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    private var canSend: Bool {
        !draft.trimmingCharacters(in: .whitespaces).isEmpty
            && !fleet.isRunning
            && fleet.modelState == .ready
    }
}

// MARK: - Turns

struct TurnView: View {
    let turn: FleetController.Turn

    // Single agent gets the full width; fleets flow in two columns.
    private var columns: [GridItem] {
        turn.agents.count == 1
            ? [GridItem(.flexible())]
            : [GridItem(.flexible(), spacing: 10), GridItem(.flexible(), spacing: 10)]
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Spacer(minLength: 48)
                Text(turn.question)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            }

            LazyVGrid(columns: columns, alignment: .leading, spacing: 10) {
                ForEach(turn.agents) { agent in
                    AgentCard(agent: agent)
                }
            }

            if let stats = turn.stats {
                Text(stats)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .center)
            }
        }
    }
}

struct AgentCard: View {
    let agent: FleetController.AgentStream

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Circle()
                    .fill(agent.done ? Color.green : Color.orange)
                    .frame(width: 7, height: 7)
                Text(agent.title)
                    .font(.caption.weight(.semibold))
                Spacer()
            }
            Group {
                if agent.text.isEmpty {
                    Text("…").foregroundStyle(.tertiary)
                } else {
                    Text(agent.text + (agent.done ? "" : " ▍"))
                }
            }
            .font(.footnote)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
        .padding(10)
        .frame(maxWidth: .infinity, minHeight: 90, alignment: .topLeading)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}

// MARK: - Sheets

struct SettingsSheet: View {
    @EnvironmentObject var fleet: FleetController
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Model") {
                    Picker("Model", selection: Binding(
                        get: { fleet.selectedModel.id },
                        set: { id in
                            if let m = FleetController.models.first(where: { $0.id == id }) {
                                fleet.select(model: m)
                            }
                        }
                    )) {
                        ForEach(FleetController.models) { m in
                            Text("\(m.family) \(m.variant)").tag(m.id)
                        }
                    }
                    .disabled(fleet.isRunning)
                }
                Section {
                    Stepper("Simultaneous agents: \(fleet.agentCount)",
                            value: $fleet.agentCount, in: 1...8)
                    Stepper("Max tokens per agent: \(fleet.maxTokensPerAgent)",
                            value: $fleet.maxTokensPerAgent, in: 64...512, step: 32)
                    Toggle("Prefix caching", isOn: $fleet.prefixCachingEnabled)
                } header: {
                    Text("Swarm")
                } footer: {
                    Text("Agents are drawn in order from: " +
                         FleetController.lenses.map(\.title).joined(separator: ", ") + ".")
                }
                Section {
                    Link("vllm-ios on GitHub",
                         destination: URL(string: "https://github.com/jonready/vllm-ios")!)
                } footer: {
                    Text("SwarmBench demonstrates continuous-batching inference on MLX: four agents share one batched decode, a cached prompt prefix, and stream concurrently.")
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { Button("Done") { dismiss() } }
        }
    }
}

struct ChatsSheet: View {
    @EnvironmentObject var fleet: FleetController
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                ForEach(fleet.chats) { chat in
                    Button {
                        fleet.select(chat: chat)
                        dismiss()
                    } label: {
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(chat.title).lineLimit(1)
                                Text("\(chat.turns.count) turn\(chat.turns.count == 1 ? "" : "s")")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                            Spacer()
                            if chat.id == fleet.currentChatID {
                                Image(systemName: "checkmark").foregroundStyle(.tint)
                            }
                        }
                    }
                    .foregroundStyle(.primary)
                }
                .onDelete { fleet.delete(at: $0) }
            }
            .navigationTitle("Chats")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { Button("Done") { dismiss() } }
        }
    }
}
