import SwiftData
import Foundation

@Model
public class Label {
    public var id: String
    public var richMessages: [RichMessage]
    
    public init(id: String, richMessages: [RichMessage] = []) {
        self.id = id
        self.richMessages = richMessages
    }
}

@Model
final public class Contact {
    public var name: String?
    public var address: String
    
    public var from: [RichMessage.Recipients]
    public var to: [RichMessage.Recipients]
    public var cc: [RichMessage.Recipients]
    public var bcc: [RichMessage.Recipients]

    public init(
        name: String? = nil,
        address: String,
        from: [RichMessage.Recipients] = [],
        to: [RichMessage.Recipients] = [],
        cc: [RichMessage.Recipients] = [],
        bcc: [RichMessage.Recipients] = []
    ) {
        self.name = name
        self.address = address
        self.from = from
        self.to = to
        self.cc = cc
        self.bcc = bcc
    }
}

@Model
final public class RichMessage {
    @Model
    final public class Recipients {
        @Relationship(inverse: \Contact.from) var from: Contact
        @Relationship(inverse: \Contact.to) var to: [Contact]
        @Relationship(inverse: \Contact.cc) var cc: [Contact]
        @Relationship(inverse: \Contact.bcc) var bcc: [Contact]
        public init(from: Contact, to: [Contact] = [], cc: [Contact] = [], bcc: [Contact] = []) {
            self.from = from
            self.to = to
            self.cc = cc
            self.bcc = bcc
        }
    }
    
    @Model
    final public class Part {
        var headers: [String: String]
        var fileName: String?
        var mimeType: String
        var partId: String?
        var body: Body?
        var parts: [Part]
        var richMessage: RichMessage?

        @Model
        final public class Body {
            var size: Int
            var data: String
            
            public init(size: Int, data: String) {
                self.size = size
                self.data = data
            }
            
            public convenience init(_ data: String) {
                self.init(size: data.count, data: data)
            }
        }
        
        public init(
            headers: [String : String] = [:],
            fileName: String? = nil,
            mimeType: String,
            partId: String? = nil,
            body: Body? = nil,
            parts: [Part] = []
        ) {
            self.headers = headers
            self.fileName = fileName
            self.mimeType = mimeType
            self.partId = partId
            self.body = body
            self.parts = parts
        }
    }
    
    nonisolated public enum Extent: Codable {
        case metadata, full
    }
    
    @Attribute(.unique) public var id: String
    var threadId: String

    public var recipients: Recipients
    public var subject: String?
    public var snippet: String?
    public var payload: Part
    @Relationship(inverse: \Label.richMessages) public var labels: [Label]
    public var historyId: String?
    public var internalDate: Date
    public var lastUpdated: Date
    public var extent: Extent
    public var structured: StructuredMessage?

    public init(
        id: String,
        threadId: String,
        recipients: Recipients,
        structured: StructuredMessage? = nil,
        subject: String? = nil,
        snippet: String? = nil,
        payload: Part,
        labels: [Label],
        historyId: String? = nil,
        internalDate: Date,
        lastUpdated: Date = Date.now,
        extent: Extent
    ) {
        self.id = id
        self.threadId = threadId
        self.recipients = recipients
        self.structured = structured
        self.subject = subject
        self.snippet = snippet
        self.payload = payload
        self.labels = labels
        self.historyId = historyId
        self.internalDate = internalDate
        self.lastUpdated = lastUpdated
        self.extent = extent
    }
}

@Model
public final class StructuredMessage {
    public init() {}
}

@Model
public final class Message: Sendable {
    @Attribute(.unique) public var id: String
    public var threadId: String
    public var createdAt: Date
    public var user: User
    public var rich: RichMessage?

    public init(
        id: String,
        threadId: String,
        createdAt: Date = Date.now,
        user: User,
        rich: RichMessage? = nil,
    ) {
        self.id = id
        self.threadId = threadId
        self.createdAt = createdAt
        self.rich = rich
        self.user = user
    }
}

extension LanguageModelMessageContent {
    init(from messagePart: RichMessage.Part, with configuration: Configuration) throws {
        guard let content = messagePart.content(), let body = content.body?.data else {
            throw LanguageModelMessageError.missingPayloadContent
        }
        
        guard content.mimeType == "text/html" else {
            try self.init(text: body, with: configuration)
            return
        }
        
        try self.init(html: body, with: configuration)
    }
}
