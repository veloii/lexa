import SwiftData
import Foundation
import SwiftUI

extension EnvironmentValues {
    @Entry var structuredMessageDefService: StructuredMessageDefService? = nil
}

@Model
class StructuredMessageDefServiceStatus {
    var currentVersion: Int?
    init(currentVersion: Int? = nil) {
        self.currentVersion = currentVersion
    }
}

protocol StructuredMessageDefService {
    func refreshStructuredMessageDefs() async throws
}

final public nonisolated class LiveStructuredMessageDefService: StructuredMessageDefService {
    private let lexaApiClient: LexaApiClient
    private let modelContext: ModelContext

    init(lexaApiClient: LexaApiClient, modelContext: ModelContext) {
        self.lexaApiClient = lexaApiClient
        self.modelContext = modelContext
    }
    
    private func getStatus() throws -> StructuredMessageDefServiceStatus? {
        try modelContext.fetch(FetchDescriptor<StructuredMessageDefServiceStatus>()).first
    }
    
    private func updateStatus(currentVersion: Int?) throws {
        if let status = try getStatus() {
            if let currentVersion {
                status.currentVersion = currentVersion
            }
        } else {
            modelContext.insert(StructuredMessageDefServiceStatus(currentVersion: currentVersion))
        }
        try modelContext.save()
    }
    
    @ModelActor
    fileprivate actor MessageInsertionActor {
        func insertMessages(from path: String, using apiClient: LexaApiClient) async throws {
            let messages = try await apiClient.perform(LexaApi.GetStructuredMessageDef(path: path))
            for message in messages {
                modelContext.insert(message)
                try modelContext.save()
            }
        }
    }
    
    func refreshStructuredMessageDefs() async throws {
        let root = try await lexaApiClient.perform(LexaApi.GetStructuredMessageDefsRoot())
        if let status = try getStatus(), let currentVersion = status.currentVersion, currentVersion >= root.version {
            return
        }
        
        let insertionActor = MessageInsertionActor(modelContainer: modelContext.container)
        try await withThrowingTaskGroup(of: Void.self) { group in
            for path in root.absolutePaths {
                group.addTask {
                    try await insertionActor.insertMessages(from: path, using: self.lexaApiClient)
                }
            }
            for try await _ in group { }
        }
        
        try updateStatus(currentVersion: root.version)
    }
}
