import SwiftUI

extension EnvironmentValues {
    @Entry var structuredMessageDefRepo: StructuredMessageDefRepo? = nil
}

public protocol StructuredMessageDefRepo {
    func listRootNodes() throws -> [StructuredMessageDefNode]
}
