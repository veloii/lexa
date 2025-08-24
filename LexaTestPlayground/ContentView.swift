import SwiftUI
import FoundationModels

struct ContentView: View {
    @State var selection: Set<Int> = [0]
    let availability = StructuredMessageServiceAvailability(from: SystemLanguageModel.default.availability)
    var available: Bool {
        availability.bool()
    }

    var body: some View {
        NavigationSplitView {
            List(selection: self.$selection) {
                Label("Test inputs", systemImage: "text.page.badge.magnifyingglass")
                    .disabled(available == false)
                    .tag(0)
            }
        } detail: {
            switch availability {
            case .available:
                SinglePromptStructuredMessageClassificationTestView()
            case .unavailable(let unavailableReason):
                Text(
                    "Apple Foundation Models are not available: \(unavailableReason.description())"
                )
            }
        }
    }
}
