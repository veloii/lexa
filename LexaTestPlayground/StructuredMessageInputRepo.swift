import SwiftUI

extension EnvironmentValues {
    @Entry var structuredMessageInputRepo: StructuredMessageInputRepo? = nil
}

@Observable
class StructuredMessageInputRepo {
    var values: [[LexaApi.LanguageModelMessage]] = []
}
