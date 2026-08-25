import SwiftUI

@main
struct FleetChatApp: App {
    @StateObject private var fleet = FleetController()

    var body: some Scene {
        WindowGroup {
            ChatView()
                .environmentObject(fleet)
        }
    }
}
