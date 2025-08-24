import Testing
import Lexa

final class ServiceTests {
    @Suite final class StructuredMessageService {
        private let structuredMessageService: FoundationModelStructuredMessageService
        
        init() throws {
            structuredMessageService = FoundationModelStructuredMessageService(configuration: try .default)
        }
        
        @Test func stripsUnnecessaryContent() throws {
            let result = try structuredMessageService.parsePartContent(
                .init(mimeType: "text/html", body: .init("<html><p>Hello, world!</p><p>This is a test.</p></html>"))
            )
            
            for substring in ["html", "p"] {
                #expect(!result.contains(substring), "Result '\(result)' contains forbidden substring: '\(substring)'")
            }
        }
        
        @Test func maintainsNecessaryContent() throws {
            let result = try structuredMessageService.parsePartContent(
                .init(mimeType: "text/html", body: .init("""
                    <a href='https://wikipedia.com'>Link 1</a>
                    <div>
                        <a href='https://example.com'>Link 2</a>
                    </div>
                """))
            )
            
            for substring in ["https://example.com", "https://wikipedia.com", "Link 1", "Link 2"] {
                #expect(result.contains(substring), "Result '\(result)' does not contain substring: '\(substring)'")
            }
        }
        
//        insertStructure
        
//        @Test(.enabled(if: structuredMessageService.availability().bool())) func maintainsNecessaryContent() throws {
//            let result = try structuredMessageService.parsePartContent(
//                .init(mimeType: "text/html", body: .init("""
//                    <a href='https://wikipedia.com'>Link 1</a>
//                    <div>
//                        <a href='https://example.com'>Link 2</a>
//                    </div>
//                """))
//            )
//            
//            for substring in ["https://example.com", "https://wikipedia.com", "Link 1", "Link 2"] {
//                #expect(result.contains(substring), "Result '\(result)' does not contain substring: '\(substring)'")
//            }
//        }
    }
}
