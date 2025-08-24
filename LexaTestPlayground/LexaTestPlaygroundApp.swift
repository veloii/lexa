import SwiftUI

@main
struct LexaTestPlaygroundApp: App {
    let lexaApiClient: LexaApiClient
    let structuredMessageDefRepo: LiveStructuredMessageDefRepo
    let structuredMessageInputRepo: StructuredMessageInputRepo
    let singlePromptTestConfig: SinglePromptStructuredMessageTestConfig

    init() {
        let lexaApiClient = LiveLexaApiClient(baseURL: URL(string: "https://lexa.aria.town")!)
        self.lexaApiClient = lexaApiClient
        self.structuredMessageDefRepo = LiveStructuredMessageDefRepo()
        self.structuredMessageInputRepo = StructuredMessageInputRepo()
        self.singlePromptTestConfig = SinglePromptStructuredMessageTestConfig()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .navigationTitle("Lexa Playground")
        }
        .environment(\.lexaApiClient, lexaApiClient)
        .environment(\.structuredMessageDefRepo, structuredMessageDefRepo)
        .environment(\.structuredMessageInputRepo, structuredMessageInputRepo)
        .environment(\.inMemoryStructuredMessageDefRepo, structuredMessageDefRepo)
        .environment(\.singlePromptTestConfig, singlePromptTestConfig)
    }
}
