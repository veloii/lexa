import SwiftData
import SwiftUI

#if DEBUG
struct SampleData: PreviewModifier {
    static func makeSharedContext() throws -> (
        LexaApiClient,
        LiveStructuredMessageDefRepo,
        StructuredMessageInputRepo,
        SinglePromptStructuredMessageTestConfig
    ) {
        let lexaApiClient = LiveLexaApiClient(baseURL: URL(string: "https://lexa.aria.town")!)
        let structuredMessageDefRepo = LiveStructuredMessageDefRepo()
        let structuredMessageInputRepo = StructuredMessageInputRepo()
        let singlePromptTestConfig = SinglePromptStructuredMessageTestConfig()
        
        return (lexaApiClient, structuredMessageDefRepo, structuredMessageInputRepo, singlePromptTestConfig)
    }
    func body(content: Content, context: (
        LexaApiClient,
        LiveStructuredMessageDefRepo,
        StructuredMessageInputRepo,
        SinglePromptStructuredMessageTestConfig)
    ) -> some View {
        content
            .environment(\.lexaApiClient, context.0)
            .environment(\.structuredMessageDefRepo, context.1)
            .environment(\.inMemoryStructuredMessageDefRepo, context.1)
            .environment(\.structuredMessageInputRepo, context.2)
            .environment(\.singlePromptTestConfig, context.3)
    }
}


@available(iOS 18.0, *)
public extension PreviewTrait where T == Preview.ViewTraits {
    @MainActor static var sampleData: Self = .modifier(SampleData())
}
#endif
