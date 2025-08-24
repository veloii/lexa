import Foundation

enum GmailApi {}

extension GmailApi {
    enum HTTPMethod: String {
        case GET, POST, PUT, DELETE, PATCH
    }
    
    nonisolated protocol Request<ResponseType>: Sendable {
        associatedtype ResponseType: Decodable, Sendable
        var path: String { get }
        var method: HTTPMethod { get }
        var queryParameters: [String: String]? { get }
        var body: Encodable? { get }
    }
    
    // MARK: User Profile Methods

    // GET /users/{userId}/profile
    struct GetUserProfileRequest: Request {
        typealias ResponseType = Profile
        let userId: String

        var path: String { "users/\(userId)/profile" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // POST /users/{userId}/stop
    struct StopUserRequest: Request {
        typealias ResponseType = UnmanagedVoid
        let userId: String

        var path: String { "users/\(userId)/stop" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // POST /users/{userId}/watch
    struct WatchUserRequest: Request {
        typealias ResponseType = WatchResponse
        let userId: String
        let watchRequest: WatchRequestOptions

        var path: String { "users/\(userId)/watch" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { watchRequest }
    }

    // MARK: Drafts Methods

    // POST /users/{userId}/drafts
    struct CreateDraftRequest: Request {
        typealias ResponseType = Draft
        let userId: String
        let draft: Draft

        var path: String { "users/\(userId)/drafts" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { draft }
    }

    // DELETE /users/{userId}/drafts/{id}
    struct DeleteDraftRequest: Request {
        typealias ResponseType = UnmanagedVoid
        let userId: String
        let id: String

        var path: String { "users/\(userId)/drafts/\(id)" }
        var method: HTTPMethod { .DELETE }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // GET /users/{userId}/drafts/{id}
    struct GetDraftRequest: Request {
        typealias ResponseType = Draft
        let userId: String
        let id: String
        let format: String?

        var path: String { "users/\(userId)/drafts/\(id)" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? {
            guard let format = format else { return nil }
            return ["format": format]
        }
        var body: Encodable? { nil }
    }

    // GET /users/{userId}/drafts
    struct ListDraftsRequest: Request {
        typealias ResponseType = ListDraftsResponse
        let userId: String
        let includeSpamTrash: Bool?
        let maxResults: Int?
        let pageToken: String?
        let query: String?

        var path: String { "users/\(userId)/drafts" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? {
            var params: [String: String] = [:]
            if let includeSpamTrash = includeSpamTrash { params["includeSpamTrash"] = String(includeSpamTrash) }
            if let maxResults = maxResults { params["maxResults"] = String(maxResults) }
            if let pageToken = pageToken { params["pageToken"] = pageToken }
            if let query = query { params["q"] = query }
            return params.isEmpty ? nil : params
        }
        var body: Encodable? { nil }
    }

    // POST /users/{userId}/drafts/send
    struct SendDraftRequest: Request {
        typealias ResponseType = Message
        let userId: String
        let draft: Draft

        var path: String { "users/\(userId)/drafts/send" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { draft }
    }

    // PUT /users/{userId}/drafts/{id}
    struct UpdateDraftRequest: Request {
        typealias ResponseType = Draft
        let userId: String
        let id: String
        let draft: Draft

        var path: String { "users/\(userId)/drafts/\(id)" }
        var method: HTTPMethod { .PUT }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { draft }
    }

    // MARK: History Methods

    // GET /users/{userId}/history
    struct ListHistoryRequest: Request {
        enum HistoryType: String {
            case messageAdded, messageDeleted, labelAdded, labelRemoved
        }
        
        typealias ResponseType = ListHistoryResponse
        let userId: String
        let historyTypes: [HistoryType]
        let labelId: String?
        let maxResults: Int?
        let pageToken: String?
        let startHistoryId: String?
        
        init(
            userId: String,
            historyTypes: [HistoryType] = [],
            labelId: String?,
            maxResults: Int?,
            pageToken: String?,
            startHistoryId: String?
        ) {
            self.userId = userId
            self.historyTypes = historyTypes
            self.labelId = labelId
            self.maxResults = maxResults
            self.pageToken = pageToken
            self.startHistoryId = startHistoryId
        }

        var path: String { "users/\(userId)/history" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? {
            var params: [String: String] = [:]
            if !historyTypes.isEmpty {
                params["[]historyTypes"] = historyTypes
                    .map(\.rawValue)
                    .joined(separator: ",")
            }
            if let labelId = labelId { params["labelId"] = labelId }
            if let maxResults = maxResults { params["maxResults"] = String(maxResults) }
            if let pageToken = pageToken { params["pageToken"] = pageToken }
            if let startHistoryId = startHistoryId { params["startHistoryId"] = startHistoryId }
            return params.isEmpty ? nil : params
        }
        var body: Encodable? { nil }
    }

    // MARK: Labels Methods

    // POST /users/{userId}/labels
    struct CreateLabelRequest: Request {
        typealias ResponseType = Label
        let userId: String
        let label: Label

        var path: String { "users/\(userId)/labels" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { label }
    }

    // DELETE /users/{userId}/labels/{id}
    struct DeleteLabelRequest: Request {
        typealias ResponseType = UnmanagedVoid
        let userId: String
        let id: String

        var path: String { "users/\(userId)/labels/\(id)" }
        var method: HTTPMethod { .DELETE }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // GET /users/{userId}/labels/{id}
    struct GetLabelRequest: Request {
        typealias ResponseType = Label
        let userId: String
        let id: String

        var path: String { "users/\(userId)/labels/\(id)" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // GET /users/{userId}/labels
    struct ListLabelsRequest: Request {
        typealias ResponseType = ListLabelsResponse
        let userId: String

        var path: String { "users/\(userId)/labels" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // PATCH /users/{userId}/labels/{id}
    struct PatchLabelRequest: Request {
        typealias ResponseType = Label
        let userId: String
        let id: String
        let label: Label

        var path: String { "users/\(userId)/labels/\(id)" }
        var method: HTTPMethod { .PATCH }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { label }
    }

    // PUT /users/{userId}/labels/{id}
    struct UpdateLabelRequest: Request {
        typealias ResponseType = Label
        let userId: String
        let id: String
        let label: Label

        var path: String { "users/\(userId)/labels/\(id)" }
        var method: HTTPMethod { .PUT }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { label }
    }

    // MARK: Messages Methods

    // POST /users/{userId}/messages/batchDelete
    struct BatchDeleteMessagesRequest: Request {
        typealias ResponseType = UnmanagedVoid
        let userId: String
        let batchDeleteMessagesRequest: BatchDeleteMessagesRequestOptions

        var path: String { "users/\(userId)/messages/batchDelete" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { batchDeleteMessagesRequest }
    }

    // POST /users/{userId}/messages/batchModify
    struct BatchModifyMessagesRequest: Request {
        typealias ResponseType = UnmanagedVoid
        let userId: String
        let batchModifyMessagesRequest: BatchModifyMessagesRequestOptions

        var path: String { "users/\(userId)/messages/batchModify" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { batchModifyMessagesRequest }
    }

    // DELETE /users/{userId}/messages/{id}
    struct DeleteMessageRequest: Request {
        typealias ResponseType = UnmanagedVoid
        let userId: String
        let id: String

        var path: String { "users/\(userId)/messages/\(id)" }
        var method: HTTPMethod { .DELETE }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // GET /users/{userId}/messages/{id}
    struct GetMessageRequest: Request {
        typealias ResponseType = Message
        let userId: String
        let id: String
        let format: Format?
        let metadataHeaders: String? = nil

        var path: String { "users/\(userId)/messages/\(id)" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? {
            var params: [String: String] = [:]
            if let format = format { params["format"] = format.rawValue }
            if let metadataHeaders = metadataHeaders { params["metadataHeaders"] = metadataHeaders }
            return params.isEmpty ? nil : params
        }
        var body: Encodable? { nil }
        
        enum Format: String {
            case minimal
            case full
            case raw
            case metadata
        }
    }

    // POST /users/{userId}/messages/import
    struct ImportMessageRequest: Request {
        typealias ResponseType = Message
        let userId: String
        let message: Message
        let deleted: Bool?
        let internalDateSource: String?
        let neverMarkSpam: Bool?
        let processForCalendar: Bool?

        var path: String { "users/\(userId)/messages/import" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? {
            var params: [String: String] = [:]
            if let deleted = deleted { params["deleted"] = String(deleted) }
            if let internalDateSource = internalDateSource { params["internalDateSource"] = internalDateSource }
            if let neverMarkSpam = neverMarkSpam { params["neverMarkSpam"] = String(neverMarkSpam) }
            if let processForCalendar = processForCalendar { params["processForCalendar"] = String(processForCalendar) }
            return params.isEmpty ? nil : params
        }
        var body: Encodable? { message }
    }

    // POST /users/{userId}/messages
    struct InsertMessageRequest: Request {
        typealias ResponseType = Message
        let userId: String
        let message: Message
        let deleted: Bool?
        let internalDateSource: String?

        var path: String { "users/\(userId)/messages" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? {
            var params: [String: String] = [:]
            if let deleted = deleted { params["deleted"] = String(deleted) }
            if let internalDateSource = internalDateSource { params["internalDateSource"] = internalDateSource }
            return params.isEmpty ? nil : params
        }
        var body: Encodable? { message }
    }

    // GET /users/{userId}/messages
    struct ListMessagesRequest: Request {
        typealias ResponseType = ListMessagesResponse
        let userId: String
        let includeSpamTrash: Bool?
        let labelIds: String?
        let maxResults: Int?
        let pageToken: String?
        let query: String?

        var path: String { "users/\(userId)/messages" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? {
            var params: [String: String] = [:]
            if let includeSpamTrash = includeSpamTrash { params["includeSpamTrash"] = String(includeSpamTrash) }
            if let labelIds = labelIds { params["labelIds"] = labelIds }
            if let maxResults = maxResults { params["maxResults"] = String(maxResults) }
            if let pageToken = pageToken { params["pageToken"] = pageToken }
            if let query = query { params["q"] = query }
            return params.isEmpty ? nil : params
        }
        var body: Encodable? { nil }
    }

    // POST /users/{userId}/messages/{id}/modify
    struct ModifyMessageRequest: Request {
        typealias ResponseType = Message
        let userId: String
        let id: String
        let modifyMessageRequest: ModifyMessageRequestOptions

        var path: String { "users/\(userId)/messages/\(id)/modify" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { modifyMessageRequest }
    }

    // POST /users/{userId}/messages/send
    struct SendMessageRequest: Request {
        typealias ResponseType = Message
        let userId: String
        let message: Message

        var path: String { "users/\(userId)/messages/send" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { message }
    }

    // POST /users/{userId}/messages/{id}/trash
    struct TrashMessageRequest: Request {
        typealias ResponseType = Message
        let userId: String
        let id: String

        var path: String { "users/\(userId)/messages/\(id)/trash" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // POST /users/{userId}/messages/{id}/untrash
    struct UntrashMessageRequest: Request {
        typealias ResponseType = Message
        let userId: String
        let id: String

        var path: String { "users/\(userId)/messages/\(id)/untrash" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // GET /users/{userId}/messages/{messageId}/attachments/{id}
    struct GetMessageAttachmentRequest: Request {
        typealias ResponseType = MessagePartBody
        let userId: String
        let messageId: String
        let id: String

        var path: String { "users/\(userId)/messages/\(messageId)/attachments/\(id)" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // MARK: Settings Methods

    // GET /users/{userId}/settings/autoForwarding
    struct GetAutoForwardingSettingsRequest: Request {
        typealias ResponseType = AutoForwarding
        let userId: String

        var path: String { "users/\(userId)/settings/autoForwarding" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // GET /users/{userId}/settings/imap
    struct GetImapSettingsRequest: Request {
        typealias ResponseType = ImapSettings
        let userId: String

        var path: String { "users/\(userId)/settings/imap" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // GET /users/{userId}/settings/language
    struct GetLanguageSettingsRequest: Request {
        typealias ResponseType = LanguageSettings
        let userId: String

        var path: String { "users/\(userId)/settings/language" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // GET /users/{userId}/settings/pop
    struct GetPopSettingsRequest: Request {
        typealias ResponseType = PopSettings
        let userId: String

        var path: String { "users/\(userId)/settings/pop" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // GET /users/{userId}/settings/vacation
    struct GetVacationSettingsRequest: Request {
        typealias ResponseType = VacationSettings
        let userId: String

        var path: String { "users/\(userId)/settings/vacation" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // PUT /users/{userId}/settings/autoForwarding
    struct UpdateAutoForwardingSettingsRequest: Request {
        typealias ResponseType = AutoForwarding
        let userId: String
        let autoForwarding: AutoForwarding

        var path: String { "users/\(userId)/settings/autoForwarding" }
        var method: HTTPMethod { .PUT }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { autoForwarding }
    }

    // PUT /users/{userId}/settings/imap
    struct UpdateImapSettingsRequest: Request {
        typealias ResponseType = ImapSettings
        let userId: String
        let imapSettings: ImapSettings

        var path: String { "users/\(userId)/settings/imap" }
        var method: HTTPMethod { .PUT }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { imapSettings }
    }

    // PUT /users/{userId}/settings/language
    struct UpdateLanguageSettingsRequest: Request {
        typealias ResponseType = LanguageSettings
        let userId: String
        let languageSettings: LanguageSettings

        var path: String { "users/\(userId)/settings/language" }
        var method: HTTPMethod { .PUT }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { languageSettings }
    }

    // PUT /users/{userId}/settings/pop
    struct UpdatePopSettingsRequest: Request {
        typealias ResponseType = PopSettings
        let userId: String
        let popSettings: PopSettings

        var path: String { "users/\(userId)/settings/pop" }
        var method: HTTPMethod { .PUT }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { popSettings }
    }

    // PUT /users/{userId}/settings/vacation
    struct UpdateVacationSettingsRequest: Request {
        typealias ResponseType = VacationSettings
        let userId: String
        let vacationSettings: VacationSettings

        var path: String { "users/\(userId)/settings/vacation" }
        var method: HTTPMethod { .PUT }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { vacationSettings }
    }

    // MARK: CSE Identity Methods

    // POST /users/{userId}/settings/cse/identities
    struct CreateCseIdentityRequest: Request {
        typealias ResponseType = CseIdentity
        let userId: String
        let cseIdentity: CseIdentity

        var path: String { "users/\(userId)/settings/cse/identities" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { cseIdentity }
    }

    // DELETE /users/{userId}/settings/cse/identities/{cseEmailAddress}
    struct DeleteCseIdentityRequest: Request {
        typealias ResponseType = UnmanagedVoid
        let userId: String
        let cseEmailAddress: String

        var path: String { "users/\(userId)/settings/cse/identities/\(cseEmailAddress)" }
        var method: HTTPMethod { .DELETE }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // GET /users/{userId}/settings/cse/identities/{cseEmailAddress}
    struct GetCseIdentityRequest: Request {
        typealias ResponseType = CseIdentity
        let userId: String
        let cseEmailAddress: String

        var path: String { "users/\(userId)/settings/cse/identities/\(cseEmailAddress)" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // GET /users/{userId}/settings/cse/identities
    struct ListCseIdentitiesRequest: Request {
        typealias ResponseType = ListCseIdentitiesResponse
        let userId: String
        let pageSize: Int?
        let pageToken: String?

        var path: String { "users/\(userId)/settings/cse/identities" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? {
            var params: [String: String] = [:]
            if let pageSize = pageSize { params["pageSize"] = String(pageSize) }
            if let pageToken = pageToken { params["pageToken"] = pageToken }
            return params.isEmpty ? nil : params
        }
        var body: Encodable? { nil }
    }

    // PATCH /users/{userId}/settings/cse/identities/{emailAddress}
    struct PatchCseIdentityRequest: Request {
        typealias ResponseType = CseIdentity
        let userId: String
        let emailAddress: String // Note: parameter name is emailAddress in path
        let cseIdentity: CseIdentity

        var path: String { "users/\(userId)/settings/cse/identities/\(emailAddress)" }
        var method: HTTPMethod { .PATCH }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { cseIdentity }
    }

    // MARK: CSE KeyPairs Methods

    // POST /users/{userId}/settings/cse/keypairs
    struct CreateCseKeyPairRequest: Request {
        typealias ResponseType = CseKeyPair
        let userId: String
        let cseKeyPair: CseKeyPair

        var path: String { "users/\(userId)/settings/cse/keypairs" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { cseKeyPair }
    }

    // POST /users/{userId}/settings/cse/keypairs/{keyPairId}:disable
    struct DisableCseKeyPairRequest: Request {
        typealias ResponseType = CseKeyPair
        let userId: String
        let keyPairId: String
        let disableRequest: DisableCseKeyPairRequestOptions
      
        var path: String { "users/\(userId)/settings/cse/keypairs/\(keyPairId):disable" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { disableRequest }
    }

    // POST /users/{userId}/settings/cse/keypairs/{keyPairId}:enable
    struct EnableCseKeyPairRequest: Request {
        typealias ResponseType = CseKeyPair
        let userId: String
        let keyPairId: String
        let enableRequest: EnableCseKeyPairRequestOptions
      
        var path: String { "users/\(userId)/settings/cse/keypairs/\(keyPairId):enable" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { enableRequest }
    }

    // GET /users/{userId}/settings/cse/keypairs/{keyPairId}
    struct GetCseKeyPairRequest: Request {
        typealias ResponseType = CseKeyPair
        let userId: String
        let keyPairId: String

        var path: String { "users/\(userId)/settings/cse/keypairs/\(keyPairId)" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // GET /users/{userId}/settings/cse/keypairs
    struct ListCseKeyPairsRequest: Request {
        typealias ResponseType = ListCseKeyPairsResponse
        let userId: String
        let pageSize: Int?
        let pageToken: String?

        var path: String { "users/\(userId)/settings/cse/keypairs" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? {
            var params: [String: String] = [:]
            if let pageSize = pageSize { params["pageSize"] = String(pageSize) }
            if let pageToken = pageToken { params["pageToken"] = pageToken }
            return params.isEmpty ? nil : params
        }
        var body: Encodable? { nil }
    }

    // POST /users/{userId}/settings/cse/keypairs/{keyPairId}:obliterate
    struct ObliterateCseKeyPairRequest: Request {
        typealias ResponseType = UnmanagedVoid
        let userId: String
        let keyPairId: String
        let obliterateRequest: ObliterateCseKeyPairRequestOptions

        var path: String { "users/\(userId)/settings/cse/keypairs/\(keyPairId):obliterate" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { obliterateRequest }
    }

    // MARK: Delegate Methods

    // POST /users/{userId}/settings/delegates
    struct CreateDelegateRequest: Request {
        typealias ResponseType = Delegate
        let userId: String
        let delegate: Delegate

        var path: String { "users/\(userId)/settings/delegates" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { delegate }
    }

    // DELETE /users/{userId}/settings/delegates/{delegateEmail}
    struct DeleteDelegateRequest: Request {
        typealias ResponseType = UnmanagedVoid
        let userId: String
        let delegateEmail: String

        var path: String { "users/\(userId)/settings/delegates/\(delegateEmail)" }
        var method: HTTPMethod { .DELETE }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // GET /users/{userId}/settings/delegates/{delegateEmail}
    struct GetDelegateRequest: Request {
        typealias ResponseType = Delegate
        let userId: String
        let delegateEmail: String

        var path: String { "users/\(userId)/settings/delegates/\(delegateEmail)" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // GET /users/{userId}/settings/delegates
    struct ListDelegatesRequest: Request {
        typealias ResponseType = ListDelegatesResponse
        let userId: String

        var path: String { "users/\(userId)/settings/delegates" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // MARK: Filter Methods

    // POST /users/{userId}/settings/filters
    struct CreateFilterRequest: Request {
        typealias ResponseType = Filter
        let userId: String
        let filter: Filter

        var path: String { "users/\(userId)/settings/filters" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { filter }
    }

    // DELETE /users/{userId}/settings/filters/{id}
    struct DeleteFilterRequest: Request {
        typealias ResponseType = UnmanagedVoid
        let userId: String
        let id: String

        var path: String { "users/\(userId)/settings/filters/\(id)" }
        var method: HTTPMethod { .DELETE }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // GET /users/{userId}/settings/filters/{id}
    struct GetFilterRequest: Request {
        typealias ResponseType = Filter
        let userId: String
        let id: String

        var path: String { "users/\(userId)/settings/filters/\(id)" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // GET /users/{userId}/settings/filters
    struct ListFiltersRequest: Request {
        typealias ResponseType = ListFiltersResponse
        let userId: String

        var path: String { "users/\(userId)/settings/filters" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // MARK: Forwarding Address Methods

    // POST /users/{userId}/settings/forwardingAddresses
    struct CreateForwardingAddressRequest: Request {
        typealias ResponseType = ForwardingAddress
        let userId: String
        let forwardingAddress: ForwardingAddress

        var path: String { "users/\(userId)/settings/forwardingAddresses" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { forwardingAddress }
    }

    // DELETE /users/{userId}/settings/forwardingAddresses/{forwardingEmail}
    struct DeleteForwardingAddressRequest: Request {
        typealias ResponseType = UnmanagedVoid
        let userId: String
        let forwardingEmail: String

        var path: String { "users/\(userId)/settings/forwardingAddresses/\(forwardingEmail)" }
        var method: HTTPMethod { .DELETE }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // GET /users/{userId}/settings/forwardingAddresses/{forwardingEmail}
    struct GetForwardingAddressRequest: Request {
        typealias ResponseType = ForwardingAddress
        let userId: String
        let forwardingEmail: String

        var path: String { "users/\(userId)/settings/forwardingAddresses/\(forwardingEmail)" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // GET /users/{userId}/settings/forwardingAddresses
    struct ListForwardingAddressesRequest: Request {
        typealias ResponseType = ListForwardingAddressesResponse
        let userId: String

        var path: String { "users/\(userId)/settings/forwardingAddresses" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // MARK: SendAs Methods

    // POST /users/{userId}/settings/sendAs
    struct CreateSendAsRequest: Request {
        typealias ResponseType = SendAs
        let userId: String
        let sendAs: SendAs

        var path: String { "users/\(userId)/settings/sendAs" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { sendAs }
    }

    // DELETE /users/{userId}/settings/sendAs/{sendAsEmail}
    struct DeleteSendAsRequest: Request {
        typealias ResponseType = UnmanagedVoid
        let userId: String
        let sendAsEmail: String

        var path: String { "users/\(userId)/settings/sendAs/\(sendAsEmail)" }
        var method: HTTPMethod { .DELETE }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // GET /users/{userId}/settings/sendAs/{sendAsEmail}
    struct GetSendAsRequest: Request {
        typealias ResponseType = SendAs
        let userId: String
        let sendAsEmail: String

        var path: String { "users/\(userId)/settings/sendAs/\(sendAsEmail)" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // GET /users/{userId}/settings/sendAs
    struct ListSendAsRequest: Request {
        typealias ResponseType = ListSendAsResponse
        let userId: String

        var path: String { "users/\(userId)/settings/sendAs" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // PATCH /users/{userId}/settings/sendAs/{sendAsEmail}
    struct PatchSendAsRequest: Request {
        typealias ResponseType = SendAs
        let userId: String
        let sendAsEmail: String
        let sendAs: SendAs

        var path: String { "users/\(userId)/settings/sendAs/\(sendAsEmail)" }
        var method: HTTPMethod { .PATCH }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { sendAs }
    }

    // PUT /users/{userId}/settings/sendAs/{sendAsEmail}
    struct UpdateSendAsRequest: Request {
        typealias ResponseType = SendAs
        let userId: String
        let sendAsEmail: String
        let sendAs: SendAs

        var path: String { "users/\(userId)/settings/sendAs/\(sendAsEmail)" }
        var method: HTTPMethod { .PUT }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { sendAs }
    }

    // POST /users/{userId}/settings/sendAs/{sendAsEmail}/verify
    struct VerifySendAsRequest: Request {
        typealias ResponseType = UnmanagedVoid
        let userId: String
        let sendAsEmail: String

        var path: String { "users/\(userId)/settings/sendAs/\(sendAsEmail)/verify" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // MARK: SendAs S/MIME Methods

    // DELETE /users/{userId}/settings/sendAs/{sendAsEmail}/smimeInfo/{id}
    struct DeleteSendAsSmimeInfoRequest: Request {
        typealias ResponseType = UnmanagedVoid
        let userId: String
        let sendAsEmail: String
        let id: String

        var path: String { "users/\(userId)/settings/sendAs/\(sendAsEmail)/smimeInfo/\(id)" }
        var method: HTTPMethod { .DELETE }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // GET /users/{userId}/settings/sendAs/{sendAsEmail}/smimeInfo/{id}
    struct GetSendAsSmimeInfoRequest: Request {
        typealias ResponseType = SmimeInfo
        let userId: String
        let sendAsEmail: String
        let id: String

        var path: String { "users/\(userId)/settings/sendAs/\(sendAsEmail)/smimeInfo/\(id)" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // POST /users/{userId}/settings/sendAs/{sendAsEmail}/smimeInfo
    struct InsertSendAsSmimeInfoRequest: Request {
        typealias ResponseType = SmimeInfo
        let userId: String
        let sendAsEmail: String
        let smimeInfo: SmimeInfo

        var path: String { "users/\(userId)/settings/sendAs/\(sendAsEmail)/smimeInfo" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { smimeInfo }
    }

    // GET /users/{userId}/settings/sendAs/{sendAsEmail}/smimeInfo
    struct ListSendAsSmimeInfosRequest: Request {
        typealias ResponseType = ListSmimeInfoResponse
        let userId: String
        let sendAsEmail: String

        var path: String { "users/\(userId)/settings/sendAs/\(sendAsEmail)/smimeInfo" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // POST /users/{userId}/settings/sendAs/{sendAsEmail}/smimeInfo/{id}/setDefault
    struct SetDefaultSendAsSmimeInfoRequest: Request {
        typealias ResponseType = UnmanagedVoid
        let userId: String
        let sendAsEmail: String
        let id: String

        var path: String { "users/\(userId)/settings/sendAs/\(sendAsEmail)/smimeInfo/\(id)/setDefault" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // MARK: Thread Methods

    // DELETE /users/{userId}/threads/{id}
    struct DeleteThreadRequest: Request {
        typealias ResponseType = UnmanagedVoid
        let userId: String
        let id: String

        var path: String { "users/\(userId)/threads/\(id)" }
        var method: HTTPMethod { .DELETE }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // GET /users/{userId}/threads/{id}
    struct GetThreadRequest: Request {
        typealias ResponseType = Thread
        let userId: String
        let id: String
        let format: String?
        let metadataHeaders: String?

        var path: String { "users/\(userId)/threads/\(id)" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? {
            var params: [String: String] = [:]
            if let format = format { params["format"] = format }
            if let metadataHeaders = metadataHeaders { params["metadataHeaders"] = metadataHeaders }
            return params.isEmpty ? nil : params
        }
        var body: Encodable? { nil }
    }

    // GET /users/{userId}/threads
    struct ListThreadsRequest: Request {
        typealias ResponseType = ListThreadsResponse
        let userId: String
        let includeSpamTrash: Bool?
        let labelIds: String?
        let maxResults: Int?
        let pageToken: String?
        let query: String?

        var path: String { "users/\(userId)/threads" }
        var method: HTTPMethod { .GET }
        var queryParameters: [String: String]? {
            var params: [String: String] = [:]
            if let includeSpamTrash = includeSpamTrash { params["includeSpamTrash"] = String(includeSpamTrash) }
            if let labelIds = labelIds { params["labelIds"] = labelIds }
            if let maxResults = maxResults { params["maxResults"] = String(maxResults) }
            if let pageToken = pageToken { params["pageToken"] = pageToken }
            if let query = query { params["q"] = query }
            return params.isEmpty ? nil : params
        }
        var body: Encodable? { nil }
    }

    // POST /users/{userId}/threads/{id}/modify
    struct ModifyThreadRequest: Request {
        typealias ResponseType = Thread
        let userId: String
        let id: String
        let modifyThreadRequest: ModifyThreadRequestOptions

        var path: String { "users/\(userId)/threads/\(id)/modify" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { modifyThreadRequest }
    }

    // POST /users/{userId}/threads/{id}/trash
    struct TrashThreadRequest: Request {
        typealias ResponseType = Thread
        let userId: String
        let id: String

        var path: String { "users/\(userId)/threads/\(id)/trash" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }

    // POST /users/{userId}/threads/{id}/untrash
    struct UntrashThreadRequest: Request {
        typealias ResponseType = Thread
        let userId: String
        let id: String

        var path: String { "users/\(userId)/threads/\(id)/untrash" }
        var method: HTTPMethod { .POST }
        var queryParameters: [String: String]? { nil }
        var body: Encodable? { nil }
    }
}

extension GmailApi {
    public nonisolated struct AutoForwarding: Codable, Sendable {
        public init (`disposition`: String?, `emailAddress`: String?, `enabled`: Bool?) {
            self.`disposition` = `disposition`
            self.`emailAddress` = `emailAddress`
            self.`enabled` = `enabled`
        }
        public var `disposition`: String?
        public var `emailAddress`: String?
        public var `enabled`: Bool?
    }
    
    public nonisolated struct BatchDeleteMessagesRequestOptions: Codable, Sendable {
        public init (`ids`: [String]?) {
            self.`ids` = `ids`
        }
        public var `ids`: [String]?
    }
    
    public nonisolated struct BatchModifyMessagesRequestOptions: Codable, Sendable {
        public init (`addLabelIds`: [String]?, `ids`: [String]?, `removeLabelIds`: [String]?) {
            self.`addLabelIds` = `addLabelIds`
            self.`ids` = `ids`
            self.`removeLabelIds` = `removeLabelIds`
        }
        public var `addLabelIds`: [String]?
        public var `ids`: [String]?
        public var `removeLabelIds`: [String]?
    }
    
    public nonisolated struct CseIdentity: Codable, Sendable {
        public init (`emailAddress`: String?, `primaryKeyPairId`: String?, `signAndEncryptKeyPairs`: SignAndEncryptKeyPairs?) {
            self.`emailAddress` = `emailAddress`
            self.`primaryKeyPairId` = `primaryKeyPairId`
            self.`signAndEncryptKeyPairs` = `signAndEncryptKeyPairs`
        }
        public var `emailAddress`: String?
        public var `primaryKeyPairId`: String?
        public var `signAndEncryptKeyPairs`: SignAndEncryptKeyPairs?
    }
    
    public nonisolated struct CseKeyPair: Codable, Sendable {
        public init (`disableTime`: String?, `enablementState`: String?, `keyPairId`: String?, `pem`: String?, `pkcs7`: String?, `privateKeyMetadata`: [CsePrivateKeyMetadata]?, `subjectEmailAddresses`: [String]?) {
            self.`disableTime` = `disableTime`
            self.`enablementState` = `enablementState`
            self.`keyPairId` = `keyPairId`
            self.`pem` = `pem`
            self.`pkcs7` = `pkcs7`
            self.`privateKeyMetadata` = `privateKeyMetadata`
            self.`subjectEmailAddresses` = `subjectEmailAddresses`
        }
        public var `disableTime`: String?
        public var `enablementState`: String?
        public var `keyPairId`: String?
        public var `pem`: String?
        public var `pkcs7`: String?
        public var `privateKeyMetadata`: [CsePrivateKeyMetadata]?
        public var `subjectEmailAddresses`: [String]?
    }
    
    public nonisolated struct CsePrivateKeyMetadata: Codable, Sendable {
        public init (`hardwareKeyMetadata`: HardwareKeyMetadata?, `kaclsKeyMetadata`: KaclsKeyMetadata?, `privateKeyMetadataId`: String?) {
            self.`hardwareKeyMetadata` = `hardwareKeyMetadata`
            self.`kaclsKeyMetadata` = `kaclsKeyMetadata`
            self.`privateKeyMetadataId` = `privateKeyMetadataId`
        }
        public var `hardwareKeyMetadata`: HardwareKeyMetadata?
        public var `kaclsKeyMetadata`: KaclsKeyMetadata?
        public var `privateKeyMetadataId`: String?
    }
    
    public nonisolated struct Delegate: Codable, Sendable {
        public init (`delegateEmail`: String?, `verificationStatus`: String?) {
            self.`delegateEmail` = `delegateEmail`
            self.`verificationStatus` = `verificationStatus`
        }
        public var `delegateEmail`: String?
        public var `verificationStatus`: String?
    }
    
    public nonisolated struct DisableCseKeyPairRequestOptions: Codable, Sendable {
        public init () {
        }
    }
    
    public nonisolated struct Draft: Codable, Sendable {
        public init (`id`: String?, `message`: Message?) {
            self.`id` = `id`
            self.`message` = `message`
        }
        public var `id`: String?
        public var `message`: Message?
    }
    
    public nonisolated struct EnableCseKeyPairRequestOptions: Codable, Sendable {
        public init () {
        }
    }
    
    public nonisolated struct Filter: Codable, Sendable {
        public init (`action`: FilterAction?, `criteria`: FilterCriteria?, `id`: String?) {
            self.`action` = `action`
            self.`criteria` = `criteria`
            self.`id` = `id`
        }
        public var `action`: FilterAction?
        public var `criteria`: FilterCriteria?
        public var `id`: String?
    }
    
    public nonisolated struct FilterAction: Codable, Sendable {
        public init (`addLabelIds`: [String]?, `forward`: String?, `removeLabelIds`: [String]?) {
            self.`addLabelIds` = `addLabelIds`
            self.`forward` = `forward`
            self.`removeLabelIds` = `removeLabelIds`
        }
        public var `addLabelIds`: [String]?
        public var `forward`: String?
        public var `removeLabelIds`: [String]?
    }
    
    public nonisolated struct FilterCriteria: Codable, Sendable {
        public init (`excludeChats`: Bool?, `from`: String?, `hasAttachment`: Bool?, `negatedQuery`: String?, `query`: String?, `size`: Int?, `sizeComparison`: String?, `subject`: String?, `to`: String?) {
            self.`excludeChats` = `excludeChats`
            self.`from` = `from`
            self.`hasAttachment` = `hasAttachment`
            self.`negatedQuery` = `negatedQuery`
            self.`query` = `query`
            self.`size` = `size`
            self.`sizeComparison` = `sizeComparison`
            self.`subject` = `subject`
            self.`to` = `to`
        }
        public var `excludeChats`: Bool?
        public var `from`: String?
        public var `hasAttachment`: Bool?
        public var `negatedQuery`: String?
        public var `query`: String?
        public var `size`: Int?
        public var `sizeComparison`: String?
        public var `subject`: String?
        public var `to`: String?
    }
    
    public nonisolated struct ForwardingAddress: Codable, Sendable {
        public init (`forwardingEmail`: String?, `verificationStatus`: String?) {
            self.`forwardingEmail` = `forwardingEmail`
            self.`verificationStatus` = `verificationStatus`
        }
        public var `forwardingEmail`: String?
        public var `verificationStatus`: String?
    }
    
    public nonisolated struct HardwareKeyMetadata: Codable, Sendable {
        public init (`description`: String?) {
            self.`description` = `description`
        }
        public var `description`: String?
    }
    
    public nonisolated struct History: Codable, Sendable {
        public init (`id`: String?, `labelsAdded`: [HistoryLabelAdded]?, `labelsRemoved`: [HistoryLabelRemoved]?, `messages`: [Message]?, `messagesAdded`: [HistoryMessageAdded]?, `messagesDeleted`: [HistoryMessageDeleted]?) {
            self.`id` = `id`
            self.`labelsAdded` = `labelsAdded`
            self.`labelsRemoved` = `labelsRemoved`
            self.`messages` = `messages`
            self.`messagesAdded` = `messagesAdded`
            self.`messagesDeleted` = `messagesDeleted`
        }
        public var `id`: String?
        public var `labelsAdded`: [HistoryLabelAdded]?
        public var `labelsRemoved`: [HistoryLabelRemoved]?
        public var `messages`: [Message]?
        public var `messagesAdded`: [HistoryMessageAdded]?
        public var `messagesDeleted`: [HistoryMessageDeleted]?
    }
    
    public nonisolated struct HistoryLabelAdded: Codable, Sendable {
        public init (`labelIds`: [String], `message`: Message) {
            self.`labelIds` = `labelIds`
            self.`message` = `message`
        }
        public var `labelIds`: [String]
        public var `message`: Message
    }
    
    public nonisolated struct HistoryLabelRemoved: Codable, Sendable {
        public init (`labelIds`: [String], `message`: Message) {
            self.`labelIds` = `labelIds`
            self.`message` = `message`
        }
        public var `labelIds`: [String]
        public var `message`: Message
    }
    
    public nonisolated struct HistoryMessageAdded: Codable, Sendable {
        public init (`message`: Message) {
            self.`message` = `message`
        }
        public var `message`: Message
    }
    
    public nonisolated struct HistoryMessageDeleted: Codable, Sendable {
        public init (`message`: Message) {
            self.`message` = `message`
        }
        public var `message`: Message
    }
    
    public nonisolated struct ImapSettings: Codable, Sendable {
        public init (`autoExpunge`: Bool?, `enabled`: Bool?, `expungeBehavior`: String?, `maxFolderSize`: Int?) {
            self.`autoExpunge` = `autoExpunge`
            self.`enabled` = `enabled`
            self.`expungeBehavior` = `expungeBehavior`
            self.`maxFolderSize` = `maxFolderSize`
        }
        public var `autoExpunge`: Bool?
        public var `enabled`: Bool?
        public var `expungeBehavior`: String?
        public var `maxFolderSize`: Int?
    }
    
    public nonisolated struct KaclsKeyMetadata: Codable, Sendable {
        public init (`kaclsData`: String?, `kaclsUri`: String?) {
            self.`kaclsData` = `kaclsData`
            self.`kaclsUri` = `kaclsUri`
        }
        public var `kaclsData`: String?
        public var `kaclsUri`: String?
    }
    
    public nonisolated struct Label: Codable, Sendable {
        public init (`color`: LabelColor?, `id`: String?, `labelListVisibility`: String?, `messageListVisibility`: String?, `messagesTotal`: Int?, `messagesUnread`: Int?, `name`: String?, `threadsTotal`: Int?, `threadsUnread`: Int?, `type`: String?) {
            self.`color` = `color`
            self.`id` = `id`
            self.`labelListVisibility` = `labelListVisibility`
            self.`messageListVisibility` = `messageListVisibility`
            self.`messagesTotal` = `messagesTotal`
            self.`messagesUnread` = `messagesUnread`
            self.`name` = `name`
            self.`threadsTotal` = `threadsTotal`
            self.`threadsUnread` = `threadsUnread`
            self.`type` = `type`
        }
        public var `color`: LabelColor?
        public var `id`: String?
        public var `labelListVisibility`: String?
        public var `messageListVisibility`: String?
        public var `messagesTotal`: Int?
        public var `messagesUnread`: Int?
        public var `name`: String?
        public var `threadsTotal`: Int?
        public var `threadsUnread`: Int?
        public var `type`: String?
    }
    
    public nonisolated struct LabelColor: Codable, Sendable {
        public init (`backgroundColor`: String?, `textColor`: String?) {
            self.`backgroundColor` = `backgroundColor`
            self.`textColor` = `textColor`
        }
        public var `backgroundColor`: String?
        public var `textColor`: String?
    }
    
    public nonisolated struct LanguageSettings: Codable, Sendable {
        public init (`displayLanguage`: String?) {
            self.`displayLanguage` = `displayLanguage`
        }
        public var `displayLanguage`: String?
    }
    
    public nonisolated struct ListCseIdentitiesResponse: Codable, Sendable {
        public init (`cseIdentities`: [CseIdentity]?, `nextPageToken`: String?) {
            self.`cseIdentities` = `cseIdentities`
            self.`nextPageToken` = `nextPageToken`
        }
        public var `cseIdentities`: [CseIdentity]?
        public var `nextPageToken`: String?
    }
    
    public nonisolated struct ListCseKeyPairsResponse: Codable, Sendable {
        public init (`cseKeyPairs`: [CseKeyPair]?, `nextPageToken`: String?) {
            self.`cseKeyPairs` = `cseKeyPairs`
            self.`nextPageToken` = `nextPageToken`
        }
        public var `cseKeyPairs`: [CseKeyPair]?
        public var `nextPageToken`: String?
    }
    
    public nonisolated struct ListDelegatesResponse: Codable, Sendable {
        public init (`delegates`: [Delegate]?) {
            self.`delegates` = `delegates`
        }
        public var `delegates`: [Delegate]?
    }
    
    public nonisolated struct ListDraftsResponse: Codable, Sendable {
        public init (`drafts`: [Draft]?, `nextPageToken`: String?, `resultSizeEstimate`: Int?) {
            self.`drafts` = `drafts`
            self.`nextPageToken` = `nextPageToken`
            self.`resultSizeEstimate` = `resultSizeEstimate`
        }
        public var `drafts`: [Draft]?
        public var `nextPageToken`: String?
        public var `resultSizeEstimate`: Int?
    }
    
    public nonisolated struct ListFiltersResponse: Codable, Sendable {
        public init (`filter`: [Filter]?) {
            self.`filter` = `filter`
        }
        public var `filter`: [Filter]?
    }
    
    public nonisolated struct ListForwardingAddressesResponse: Codable, Sendable {
        public init (`forwardingAddresses`: [ForwardingAddress]?) {
            self.`forwardingAddresses` = `forwardingAddresses`
        }
        public var `forwardingAddresses`: [ForwardingAddress]?
    }
    
    public nonisolated struct ListHistoryResponse: Codable, Sendable {
        public init (`history`: [History]?, `historyId`: String?, `nextPageToken`: String?) {
            self.`history` = `history`
            self.`historyId` = `historyId`
            self.`nextPageToken` = `nextPageToken`
        }
        public var `history`: [History]?
        public var `historyId`: String?
        public var `nextPageToken`: String?
    }
    
    public nonisolated struct ListLabelsResponse: Codable, Sendable {
        public init (`labels`: [Label]?) {
            self.`labels` = `labels`
        }
        public var `labels`: [Label]?
    }
    
    public nonisolated struct ListMessagesResponse: Codable, Sendable {
        public init (`messages`: [Message], `nextPageToken`: String?, `resultSizeEstimate`: Int?) {
            self.`messages` = `messages`
            self.`nextPageToken` = `nextPageToken`
            self.`resultSizeEstimate` = `resultSizeEstimate`
        }
        public var `messages`: [Message]
        public var `nextPageToken`: String?
        public var `resultSizeEstimate`: Int?
    }
    
    public nonisolated struct ListSendAsResponse: Codable, Sendable {
        public init (`sendAs`: [SendAs]?) {
            self.`sendAs` = `sendAs`
        }
        public var `sendAs`: [SendAs]?
    }
    
    public nonisolated struct ListSmimeInfoResponse: Codable, Sendable {
        public init (`smimeInfo`: [SmimeInfo]?) {
            self.`smimeInfo` = `smimeInfo`
        }
        public var `smimeInfo`: [SmimeInfo]?
    }
    
    public nonisolated struct ListThreadsResponse: Codable, Sendable {
        public init (`nextPageToken`: String?, `resultSizeEstimate`: Int?, `threads`: [Thread]?) {
            self.`nextPageToken` = `nextPageToken`
            self.`resultSizeEstimate` = `resultSizeEstimate`
            self.`threads` = `threads`
        }
        public var `nextPageToken`: String?
        public var `resultSizeEstimate`: Int?
        public var `threads`: [Thread]?
    }
    
    public nonisolated struct Message: Codable, Sendable {
        public init (`historyId`: String?, `id`: String, `internalDate`: String?, `labelIds`: [String]?, `payload`: MessagePart?, `raw`: String?, `sizeEstimate`: Int?, `snippet`: String?, `threadId`: String) {
            self.`historyId` = `historyId`
            self.`id` = `id`
            self.`internalDate` = `internalDate`
            self.`labelIds` = `labelIds`
            self.`payload` = `payload`
            self.`raw` = `raw`
            self.`sizeEstimate` = `sizeEstimate`
            self.`snippet` = `snippet`
            self.`threadId` = `threadId`
        }
        public var `historyId`: String?
        public var `id`: String
        public var `internalDate`: String?
        public var `labelIds`: [String]?
        public var `payload`: MessagePart?
        public var `raw`: String?
        public var `sizeEstimate`: Int?
        public var `snippet`: String?
        public var `threadId`: String
    }
    
    public nonisolated struct MessagePart: Codable, Sendable {
        public init (`body`: MessagePartBody?, `filename`: String?, `headers`: [MessagePartHeader]?, `mimeType`: String?, `partId`: String?, `parts`: [MessagePart]?) {
            self.`body` = `body`
            self.`filename` = `filename`
            self.`headers` = `headers`
            self.`mimeType` = `mimeType`
            self.`partId` = `partId`
            self.`parts` = `parts`
        }
        public var `body`: MessagePartBody?
        public var `filename`: String?
        public var `headers`: [MessagePartHeader]?
        public var `mimeType`: String?
        public var `partId`: String?
        public var `parts`: [MessagePart]?
    }
    
    public nonisolated struct MessagePartBody: Codable, Sendable {
        public init (`attachmentId`: String?, `data`: String?, `size`: Int?) {
            self.`attachmentId` = `attachmentId`
            self.`data` = `data`
            self.`size` = `size`
        }
        public var `attachmentId`: String?
        public var `data`: String?
        public var `size`: Int?
    }
    
    public nonisolated struct MessagePartHeader: Codable, Sendable {
        public init (`name`: String?, `value`: String?) {
            self.`name` = `name`
            self.`value` = `value`
        }
        public var `name`: String?
        public var `value`: String?
    }
    
    public nonisolated struct ModifyMessageRequestOptions: Codable, Sendable {
        public init (`addLabelIds`: [String]?, `removeLabelIds`: [String]?) {
            self.`addLabelIds` = `addLabelIds`
            self.`removeLabelIds` = `removeLabelIds`
        }
        public var `addLabelIds`: [String]?
        public var `removeLabelIds`: [String]?
    }
    
    public nonisolated struct ModifyThreadRequestOptions: Codable, Sendable {
        public init (`addLabelIds`: [String]?, `removeLabelIds`: [String]?) {
            self.`addLabelIds` = `addLabelIds`
            self.`removeLabelIds` = `removeLabelIds`
        }
        public var `addLabelIds`: [String]?
        public var `removeLabelIds`: [String]?
    }
    
    public nonisolated struct ObliterateCseKeyPairRequestOptions: Codable, Sendable {
        public init () {
        }
    }
    
    public nonisolated struct PopSettings: Codable, Sendable {
        public init (`accessWindow`: String?, `disposition`: String?) {
            self.`accessWindow` = `accessWindow`
            self.`disposition` = `disposition`
        }
        public var `accessWindow`: String?
        public var `disposition`: String?
    }
    
    public nonisolated struct Profile: Codable, Sendable {
        public init (`emailAddress`: String?, `historyId`: String?, `messagesTotal`: Int?, `threadsTotal`: Int?) {
            self.`emailAddress` = `emailAddress`
            self.`historyId` = `historyId`
            self.`messagesTotal` = `messagesTotal`
            self.`threadsTotal` = `threadsTotal`
        }
        public let `emailAddress`: String?
        public let `historyId`: String?
        public let `messagesTotal`: Int?
        public let `threadsTotal`: Int?
    }
    
    public nonisolated struct SendAs: Codable, Sendable {
        public init (`displayName`: String?, `isDefault`: Bool?, `isPrimary`: Bool?, `replyToAddress`: String?, `sendAsEmail`: String?, `signature`: String?, `smtpMsa`: SmtpMsa?, `treatAsAlias`: Bool?, `verificationStatus`: String?) {
            self.`displayName` = `displayName`
            self.`isDefault` = `isDefault`
            self.`isPrimary` = `isPrimary`
            self.`replyToAddress` = `replyToAddress`
            self.`sendAsEmail` = `sendAsEmail`
            self.`signature` = `signature`
            self.`smtpMsa` = `smtpMsa`
            self.`treatAsAlias` = `treatAsAlias`
            self.`verificationStatus` = `verificationStatus`
        }
        public var `displayName`: String?
        public var `isDefault`: Bool?
        public var `isPrimary`: Bool?
        public var `replyToAddress`: String?
        public var `sendAsEmail`: String?
        public var `signature`: String?
        public var `smtpMsa`: SmtpMsa?
        public var `treatAsAlias`: Bool?
        public var `verificationStatus`: String?
    }
    
    public nonisolated struct SignAndEncryptKeyPairs: Codable, Sendable {
        public init (`encryptionKeyPairId`: String?, `signingKeyPairId`: String?) {
            self.`encryptionKeyPairId` = `encryptionKeyPairId`
            self.`signingKeyPairId` = `signingKeyPairId`
        }
        public var `encryptionKeyPairId`: String?
        public var `signingKeyPairId`: String?
    }
    
    public nonisolated struct SmimeInfo: Codable, Sendable {
        public init (`encryptedKeyPassword`: String?, `expiration`: String?, `id`: String?, `isDefault`: Bool?, `issuerCn`: String?, `pem`: String?, `pkcs12`: String?) {
            self.`encryptedKeyPassword` = `encryptedKeyPassword`
            self.`expiration` = `expiration`
            self.`id` = `id`
            self.`isDefault` = `isDefault`
            self.`issuerCn` = `issuerCn`
            self.`pem` = `pem`
            self.`pkcs12` = `pkcs12`
        }
        public var `encryptedKeyPassword`: String?
        public var `expiration`: String?
        public var `id`: String?
        public var `isDefault`: Bool?
        public var `issuerCn`: String?
        public var `pem`: String?
        public var `pkcs12`: String?
    }
    
    public nonisolated struct SmtpMsa: Codable, Sendable {
        public init (`host`: String?, `password`: String?, `port`: Int?, `securityMode`: String?, `username`: String?) {
            self.`host` = `host`
            self.`password` = `password`
            self.`port` = `port`
            self.`securityMode` = `securityMode`
            self.`username` = `username`
        }
        public var `host`: String?
        public var `password`: String?
        public var `port`: Int?
        public var `securityMode`: String?
        public var `username`: String?
    }
    
    public nonisolated struct Thread: Codable, Sendable {
        public init (`historyId`: String?, `id`: String?, `messages`: [Message]?, `snippet`: String?) {
            self.`historyId` = `historyId`
            self.`id` = `id`
            self.`messages` = `messages`
            self.`snippet` = `snippet`
        }
        public var `historyId`: String?
        public var `id`: String?
        public var `messages`: [Message]?
        public var `snippet`: String?
    }
    
    public nonisolated struct VacationSettings: Codable, Sendable {
        public init (`enableAutoReply`: Bool?, `endTime`: String?, `responseBodyHtml`: String?, `responseBodyPlainText`: String?, `responseSubject`: String?, `restrictToContacts`: Bool?, `restrictToDomain`: Bool?, `startTime`: String?) {
            self.`enableAutoReply` = `enableAutoReply`
            self.`endTime` = `endTime`
            self.`responseBodyHtml` = `responseBodyHtml`
            self.`responseBodyPlainText` = `responseBodyPlainText`
            self.`responseSubject` = `responseSubject`
            self.`restrictToContacts` = `restrictToContacts`
            self.`restrictToDomain` = `restrictToDomain`
            self.`startTime` = `startTime`
        }
        public var `enableAutoReply`: Bool?
        public var `endTime`: String?
        public var `responseBodyHtml`: String?
        public var `responseBodyPlainText`: String?
        public var `responseSubject`: String?
        public var `restrictToContacts`: Bool?
        public var `restrictToDomain`: Bool?
        public var `startTime`: String?
    }
    
    public nonisolated struct WatchRequestOptions: Codable, Sendable {
        public init (`labelFilterAction`: String?, `labelFilterBehavior`: String?, `labelIds`: [String]?, `topicName`: String?) {
            self.`labelFilterAction` = `labelFilterAction`
            self.`labelFilterBehavior` = `labelFilterBehavior`
            self.`labelIds` = `labelIds`
            self.`topicName` = `topicName`
        }
        public var `labelFilterAction`: String?
        public var `labelFilterBehavior`: String?
        public var `labelIds`: [String]?
        public var `topicName`: String?
    }
    
    public nonisolated struct WatchResponse: Codable, Sendable {
        public init (`expiration`: String?, `historyId`: String?) {
            self.`expiration` = `expiration`
            self.`historyId` = `historyId`
        }
        public var `expiration`: String?
        public var `historyId`: String?
    }
}

