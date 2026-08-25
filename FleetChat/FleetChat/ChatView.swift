import SwiftUI

struct ChatView: View {
    @EnvironmentObject var fleet: FleetController
    @State private var draft = ""
    @FocusState private var composerFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            header
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 20) {
                        if fleet.turns.isEmpty {
                            emptyState
                        }
                        ForEach(fleet.turns) { turn in
                            TurnView(turn: turn)
                                .id(turn.id)
                        }
                        Color.clear.frame(height: 1).id("bottom")
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 12)
                }
                .onChange(of: fleet.turns.last?.agents.map(\.text.count).reduce(0, +) ?? 0) {
                    withAnimation(.easeOut(duration: 0.15)) {
                        proxy.scrollTo("bottom", anchor: .bottom)
                    }
                }
            }
            composer
        }
        .background(Color(.systemBackground))
        .onAppear { fleet.loadModelIfNeeded() }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Text("FleetChat")
                .font(.headline)
            Spacer()
            switch fleet.modelState {
            case .idle, .loading:
                HStack(spacing: 6) {
                    ProgressView().controlSize(.mini)
                    if case .loading(let msg) = fleet.modelState {
                        Text(msg).font(.caption2).foregroundStyle(.secondary)
                    }
                }
            case .ready:
                Label("Qwen3.5 0.8B · on-device", systemImage: "cpu")
                    .font(.caption2).foregroundStyle(.secondary)
                    .labelStyle(.titleAndIcon)
            case .failed(let msg):
                Text(msg).font(.caption2).foregroundStyle(.red).lineLimit(1)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.bar)
    }

    private var emptyState: some View {
        VStack(spacing: 10) {
            Image(systemName: "point.3.connected.trianglepath.dotted")
                .font(.system(size: 40))
                .foregroundStyle(.tertiary)
            Text("Ask anything.\nFour on-device agents answer it in parallel.")
                .multilineTextAlignment(.center)
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 120)
    }

    private var composer: some View {
        HStack(alignment: .bottom, spacing: 10) {
            TextField("Message", text: $draft, axis: .vertical)
                .lineLimit(1...5)
                .focused($composerFocused)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(
                    RoundedRectangle(cornerRadius: 22, style: .continuous)
                        .fill(Color(.secondarySystemBackground))
                )
            Button {
                let text = draft
                draft = ""
                fleet.send(text)
            } label: {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.system(size: 30))
                    .foregroundStyle(canSend ? Color.accentColor : Color(.systemGray3))
            }
            .disabled(!canSend)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(.bar)
    }

    private var canSend: Bool {
        !draft.trimmingCharacters(in: .whitespaces).isEmpty
            && !fleet.isRunning
            && fleet.modelState == .ready
    }
}

struct TurnView: View {
    let turn: FleetController.Turn

    private let columns = [GridItem(.flexible(), spacing: 10), GridItem(.flexible(), spacing: 10)]

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            // User message, ChatGPT-style right-aligned bubble.
            HStack {
                Spacer(minLength: 48)
                Text(turn.question)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .fill(Color(.secondarySystemBackground))
                    )
            }

            // The fleet: four agent cards streaming concurrently.
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
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color(.secondarySystemBackground).opacity(0.6))
                .overlay(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .strokeBorder(Color(.separator).opacity(0.5), lineWidth: 0.5)
                )
        )
    }
}
