import SwiftUI
import SwiftData

struct ContentView: View {
    @Environment(\.modelContext) private var modelContext

    var body: some View {
        InboxView()
    }
}

#Preview {
    ContentView()
        .modelContainer(for: User.self, inMemory: true)
}

