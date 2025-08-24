import SwiftUI
import SwiftData

@main
struct MainEntryPoint {
    static func main() {
        guard isTesting() else {
            NoopApp.main()
            return
        }
        
        LexaApp.main()
    }
    
    private static func isTesting() -> Bool {
        return NSClassFromString("XCTestCase") == nil
    }
}

struct NoopApp: App {
    var body: some Scene {
        WindowGroup {
        }
    }
}

struct LexaApp: App {
    var sharedModelContainer: ModelContainer = {
        let schema = Schema([
            User.self,
            Label.self,
            Contact.self,
            RichMessage.self,
            RichMessage.Recipients.self,
            RichMessage.Part.self,
            RichMessage.Part.Body.self,
            Message.self,
            StructuredMessage.self,
            StructuredMessageDefNode.self,
            StructuredMessageDefServiceStatus.self
        ])
        let modelConfiguration = ModelConfiguration(schema: schema, isStoredInMemoryOnly: false)

        do {
            return try ModelContainer(for: schema, configurations: [modelConfiguration])
        } catch {
            fatalError("Could not create ModelContainer: \(error)")
        }
    }()
    
    let authenticationService: AuthenticationService
    let messageService: MessageService
    let gmailApiClient: GmailApiClient
    let structuredMessageDefService: StructuredMessageDefService
    let lexaApiClient: LexaApiClient

    init() {
        let modelContext = sharedModelContainer.mainContext;

        let liveAuthenticationService = LiveAuthenticationService(
            modelContext: modelContext
        );
        let gmailApiClient = LiveGmailApiClient(
            baseURL: URL(string: "https://gmail.googleapis.com/gmail/v1/")!,
            tokenProvider: liveAuthenticationService
        )
        let lexaApiClient = LiveLexaApiClient(baseURL: URL(string: "https://lexa.aria.town")!)
        
        self.authenticationService = liveAuthenticationService
        self.messageService = LiveMessageService(
            authenticationService: liveAuthenticationService,
            modelContext: modelContext,
            apiClient: gmailApiClient,
            configuration: .init(
                stale: 60 * 5,
                resultBatchSize: 50,
                maxImportCount: 1000
            )
        )
        self.structuredMessageDefService = LiveStructuredMessageDefService(
            lexaApiClient: lexaApiClient,
            modelContext: modelContext
        )
        self.gmailApiClient = gmailApiClient
        self.lexaApiClient = lexaApiClient
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .onOpenURL(perform: authenticationService.handleURL)
                .task {
                    let _ = try? await authenticationService.setup()
                }
        }
        .modelContainer(sharedModelContainer)
        .environment(\.authenticationService, authenticationService)
        .environment(\.messageService, messageService)
        .environment(\.gmailApiClient, gmailApiClient)
        .environment(\.lexaApiClient, lexaApiClient)
        .environment(\.structuredMessageDefService, structuredMessageDefService)
    }
}
