import SwiftUI

@main
struct SwarmBenchApp: App {
    @StateObject private var fleet = FleetController()

    var body: some Scene {
        WindowGroup {
            ChatView()
                .environmentObject(fleet)
        }
    }
}
