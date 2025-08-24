import Foundation
import SwiftData
import SwiftUI

protocol MessageService {
    func onReachedMessagesEnd() async throws
    func sync() async throws
}

final class NoopMessageService: MessageService {
    func onReachedMessagesEnd() async throws {}
    func sync() async throws {}
}

final class LiveMessageService: MessageService {
    private var authenticationService: AuthenticationService
    private var modelContext: ModelContext
    private var apiClient: GmailApiClient
    private var configuration: Configuration
    
    init(
        authenticationService: AuthenticationService,
        modelContext: ModelContext,
        apiClient: GmailApiClient,
        configuration: Configuration
        
    ) {
        self.authenticationService = authenticationService
        self.modelContext = modelContext
        self.apiClient = apiClient
        self.configuration = configuration
    }
    
    // MARK: - Public API
    func onReachedMessagesEnd() async throws {
    }
    
    func sync() async throws {
        guard let user = self.authenticationService.activeUser else {
            return
        }
        
        if let historyId = try self.getLatestHistoryId(user: user) {
            do {
                try await self.performPartialSync(fromHistoryId: historyId)
                return
            } catch let error as LiveMessageServiceError {
                switch error {
                case .missingMessageReferencedInHistory:
                    break
                }
            } catch let error as NetworkError {
                switch error {
                case .clientError(let statusCode, _):
                    if statusCode != 404 {
                        throw error
                    }
                default:
                    throw error
                }
            }
        }
        
        try await self.performFullSync()
    }
    
    // MARK: - Full Sync
    
    private func performFullSync() async throws {
        guard let user = self.authenticationService.activeUser else {
            return
        }
        
        user.lastHistoryToken = nil
        
        var pageToken: String? = nil
        let userId = user.id
        try self.modelContext.fetch(FetchDescriptor<Message>(
            predicate: #Predicate { model in
                model.user.id == userId
            }
        )).forEach(self.modelContext.delete)
        
        var importCount = 0
        
        repeat {
            let response = try await apiClient.perform(GmailApi.ListMessagesRequest(
                userId: user.id,
                includeSpamTrash: false,
                labelIds: nil,
                maxResults: self.configuration.resultBatchSize,
                pageToken: pageToken,
                query: nil
            ))
            
            importCount += response.messages.count
            
            pageToken = response.nextPageToken
            try await insertNewMessages(response.messages, user: user)
        } while pageToken != nil && importCount < configuration.maxImportCount
        
        try self.modelContext.save()
    }
    
    // MARK: - Partial Sync
    
    private func getLatestHistoryId(user: User) throws -> String? {
        if let historyId = user.lastHistoryToken {
            return historyId
        }
        
        let userId = user.id;
        var descriptor = FetchDescriptor<Message>(
            predicate: #Predicate {
                $0.user.id == userId && $0.rich != nil
            },
            sortBy: [SortDescriptor(\.rich?.internalDate, order: .reverse)]
        )
        descriptor.fetchLimit = 1
        
        let results = try modelContext.fetch(descriptor)
        return results.first?.rich?.historyId
    }
    
    private func performHistoryEntry(_ history: GmailApi.History, user: User) async throws {
        for messageDeleted in history.messagesDeleted ?? [] {
            let message = try getMessage(byId: messageDeleted.message.id)
            guard let message else {
                throw LiveMessageServiceError.missingMessageReferencedInHistory
            }
            self.modelContext.delete(message)
        }
        
        for messageAdded in history.messagesAdded ?? [] {
            try await insertNewMessage(messageAdded.message, user: user)
        }
        
        for labelsRemovedEntry in history.labelsRemoved ?? [] {
            let rich = try await getMessageRich(byId: labelsRemovedEntry.message.id)
            guard let rich else {
                throw LiveMessageServiceError.missingMessageReferencedInHistory
            }
            rich.labels.removeAll(where: { labelsRemovedEntry.labelIds.contains($0.id) })
        }
        
        for labelsAddedEntry in history.labelsAdded ?? [] {
            let rich = try await getMessageRich(byId: labelsAddedEntry.message.id)
            guard let rich else {
                throw LiveMessageServiceError.missingMessageReferencedInHistory
            }
            rich.labels.append(contentsOf: try [Label](
                from: labelsAddedEntry.labelIds,
                using: self.modelContext
            ))
        }
    }
    
    private func performPartialSync(fromHistoryId historyId: String) async throws {
        guard let user = self.authenticationService.activeUser else {
            return
        }
        
        var pageToken: String? = nil
        
        repeat {
            let response = try await apiClient.perform(GmailApi.ListHistoryRequest(
                userId: user.id,
                historyTypes: [.messageAdded, .messageDeleted, .labelAdded, .labelRemoved],
                labelId: nil,
                maxResults: self.configuration.resultBatchSize,
                pageToken: pageToken,
                startHistoryId: historyId
            ))
            
            user.lastHistoryToken = response.historyId
            pageToken = response.nextPageToken
            
            if let history = response.history {
                for historyEntry in history {
                    try await performHistoryEntry(historyEntry, user: user)
                }
            }
        } while pageToken != nil
        
        try self.modelContext.save()
    }
    
    // MARK: - Messages
    
    private func getMessage(byId messageId: String) throws -> Message? {
        try self.modelContext.fetch(FetchDescriptor<Message>(
            predicate: #Predicate { model in
                model.id == messageId
            }
        )).first
    }
    
    private func getMessageRich(byId messageId: String) async throws -> RichMessage? {
        guard let first = try getMessage(byId: messageId) else { return nil }
        if first.rich == nil {
            try await updateRich(for: first, extent: .metadata)
        }
        return first.rich!
    }
    
    private func isRichStale(_ rich: RichMessage) -> Bool {
        let timeInterval = rich.lastUpdated.timeIntervalSinceNow * -1
        return timeInterval > self.configuration.stale
    }
    
    private func updateRich(for message: Message, extent: RichMessage.Extent) async throws {
        let response = try await apiClient.perform(GmailApi.GetMessageRequest(
            userId: message.user.id,
            id: message.id,
            format: extent.gmailApiFormat
        ))
        if let rich = message.rich {
            try rich.update(from: response, extent: extent, using: modelContext)
        } else {
            message.rich = try RichMessage(from: response, extent: .metadata, using: modelContext)
        }
    }
    
    private func updateRich(for messages: [Message], extent: RichMessage.Extent) async throws {
        for message in messages {
            try await updateRich(for: message, extent: extent)
        }
    }
    
    private func insertNewMessage(_ message: GmailApi.Message, user: User) async throws {
        try await self.insertNewMessages([message], user: user)
    }
    private func insertNewMessages(_ messages: [GmailApi.Message], user: User) async throws {
        var insertedMessages = [Message]()
        
        for message in messages {
            let insertedMessage = try Message(from: message, user: user)
            self.modelContext.insert(insertedMessage)
            insertedMessages.append(insertedMessage)
        }
        
        try await updateRich(for: insertedMessages, extent: .metadata)
    }
}

enum LiveMessageServiceError: Error {
    case missingMessageReferencedInHistory
}

extension EnvironmentValues {
    @Entry var messageService: MessageService? = nil
}

extension LiveMessageService {
    struct Configuration {
        var stale: TimeInterval
        var resultBatchSize: Int
        var maxImportCount: Int
    }
}

extension RichMessage.Extent {
    var gmailApiFormat: GmailApi.GetMessageRequest.Format {
        switch self {
        case .full:
            return .full
        case .metadata:
            return .metadata
        }
    }
}

