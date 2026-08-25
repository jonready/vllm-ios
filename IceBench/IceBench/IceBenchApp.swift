import SwiftUI

@main
struct IceBenchApp: App {
    @StateObject private var bench = BenchController()
    @StateObject private var models = ModelManager()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(bench)
                .environmentObject(models)
        }
    }
}
