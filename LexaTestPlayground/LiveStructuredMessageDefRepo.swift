import SwiftUI

extension EnvironmentValues {
    @Entry var inMemoryStructuredMessageDefRepo: LiveStructuredMessageDefRepo? = nil
}

@Observable
class LiveStructuredMessageDefRepo: StructuredMessageDefRepo {
    var all: [StructuredMessageDefNode]
    init(all: [StructuredMessageDefNode] = []) {
        self.all = all
    }
    
    func listRootNodes() throws -> [StructuredMessageDefNode] {
        all.filter { $0.parent == nil }
    }
}
