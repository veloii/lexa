import Foundation
import SwiftData

public struct LiveStructuredMessageDefRepo: StructuredMessageDefRepo {
    private let modelContext: ModelContext
    
    init(modelContext: ModelContext) {
        self.modelContext = modelContext
    }
    
    public func listRootNodes() throws -> [StructuredMessageDefNode] {
        try modelContext.fetch(FetchDescriptor<StructuredMessageDefNode>(
            predicate: #Predicate { $0.parent == nil }
        ))
    }
}
