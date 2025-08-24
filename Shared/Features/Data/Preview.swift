import SwiftData
import SwiftUI

#if DEBUG
class PreviewContainer {
    static let shared: ModelContainer = {
        let config = ModelConfiguration(isStoredInMemoryOnly: true)
        let container = try! ModelContainer(
            for: User.self,
            Label.self,
            Contact.self,
            RichMessage.self,
            RichMessage.Recipients.self,
            RichMessage.Part.self,
            RichMessage.Part.Body.self,
            Message.self,
            configurations: config
        )
        
        let context = container.mainContext
        let userCount = (try? context.fetchCount(FetchDescriptor<User>())) ?? 0
        
        if userCount == 0 {
            SampleData.createSampleData(into: context)
        }
        
        return container
    }()
    
    static var context: ModelContext { shared.mainContext }
}

struct SampleData: PreviewModifier {
    static func makeSharedContext() throws -> (
        ModelContainer,
        AuthenticationService,
        MessageService
    ) {
        let container = PreviewContainer.shared
        
        return (
            container,
            MockAuthenticationService(modelContext: container.mainContext),
            NoopMessageService()
        )
    }
    func body(content: Content, context: (ModelContainer, AuthenticationService, MessageService)) -> some View {
        content
            .modelContainer(context.0)
            .environment(\.authenticationService, context.1)
            .environment(\.messageService, context.2)
            .onOpenURL(perform: context.1.handleURL)
            .task {
                let _ = try? await context.1.setup()
            }
    }
    
    static func createSampleData(into modelContext: ModelContext) {
        Task { @MainActor in
            let sampleData: [any PersistentModel] = User.samples + Message.samples
            sampleData.forEach {
                modelContext.insert($0)
            }
            try? modelContext.save()
        }
    }
}


@available(iOS 18.0, *)
public extension PreviewTrait where T == Preview.ViewTraits {
    @MainActor static var sampleData: Self = .modifier(SampleData())
}
#endif
