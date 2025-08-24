import SwiftData
import KeychainAccess
import Foundation
import GoogleSignIn

nonisolated public struct Token: Codable {
  var accessToken: String?
  var refreshToken: String?
  var idToken: String?
  var tokenExpirationDate: Date?
  
  var isTokenExpired: Bool {
      guard let expirationDate = tokenExpirationDate else { return true }
      return Date() > expirationDate
  }
    
    static func from(response: TokenResponse) -> Token {
        Token(
            accessToken: response.accessToken,
            refreshToken: response.refreshToken,
            tokenExpirationDate: response.expirationDate
        )
    }
}

@Model
final public class User: Sendable {
    public var id: String
    public var email: String
    public var sessionId: String
    public var name: String
    public var profileImageURL: String?
    public var lastHistoryToken: String?
    public var createdAt: Date
    @Relationship(deleteRule: .cascade, inverse: \Message.user) public var messages: [Message]

    public var keychain: Keychain {
    Keychain(service: "io.veloi.lexa.token")
  }
  
    public func setToken(_ token: Token) throws {
    guard let data = try? JSONEncoder().encode(token) else {
      return
    }
    try keychain.set(data, key: self.id);
  }
  
    public func getToken() throws -> Token? {
    guard let data = try? keychain.getData(self.id) else {
      return nil
    }
    
    guard let token = try? JSONDecoder().decode(Token.self, from: data) else {
      return nil
    }
    
    return token
  }
    
    public init(
        id: String,
        sessionId: String,
        name: String,
        email: String,
        profileImageURL: String? = nil,
        createdAt: Date,
        messages: [Message] = [],
        token: Token
    ) throws {
        self.id = id
        self.email = email
        self.sessionId = sessionId
        self.name = name
        self.profileImageURL = profileImageURL
        self.createdAt = createdAt
        self.messages = messages
        try self.setToken(token)
    }

    public init(
        id: String,
        sessionId: String,
        name: String,
        email: String,
        profileImageURL: String? = nil,
        messages: [Message] = [],
        createdAt: Date
    ) {
        self.id = id
        self.email = email
        self.sessionId = sessionId
        self.name = name
        self.profileImageURL = profileImageURL
        self.createdAt = createdAt
        self.messages = messages
    }
    
    static func from(googleUser: GIDGoogleUser) throws -> User {
        guard let userId = googleUser.userID else {
            throw GoogleSignInError.missingUserId
        }
        
        let sessionId = UUID().uuidString
        
        let token = Token(
            accessToken: googleUser.accessToken.tokenString,
            refreshToken: googleUser.refreshToken.tokenString,
            idToken: googleUser.idToken?.tokenString,
            tokenExpirationDate: googleUser.accessToken.expirationDate
        )
                          
        return try User(
            id: userId,
            sessionId: sessionId,
            name: googleUser.profile?.name ?? "unknown username",
            email: googleUser.profile?.email ?? "unknown email",
            profileImageURL: googleUser.profile?.imageURL(withDimension: 100)?.absoluteString,
            createdAt: Date(),
            token: token
        )
    }
}

extension User {
    static let samples = [
        User(
            id: "123",
            sessionId: "123",
            name: "John Doe",
            email: "john@doe.com",
            createdAt: Date.now
        ),
        User(
            id: "234",
            sessionId: "234",
            name: "Jane Doe",
            email: "jane@doe.com",
            createdAt: Date.now
        ),
        User(
            id: "345",
            sessionId: "345",
            name: "X Doe",
            email: "x@doe.com",
            createdAt: Date.now
        ),
    ]
}

extension Contact {
    static let samples = [
        Contact(name: "John", address: "john@doe.com")
    ]
}

extension Label {
    static let primary = Label(id: "PRIMARY")
    static let promotions = Label(id: "PROMOTIONS")
    static let important = Label(id: "IMPORTANT")
}

extension Message {
    static let samples = [
        // Conservative Party
        Message(
            id: "001",
            threadId: "tory-shambles",
            user: User.samples.first!,
            rich: RichMessage(
                id: "001",
                threadId: "tory-shambles",
                recipients: RichMessage.Recipients(
                    from: Contact(address: "rishi.sunak@conservatives.com"),
                    to: [Contact.samples.first!],
                    cc: [],
                    bcc: []
                ),
                subject: "RE: Economic Growth Strategy",
                snippet: "We've tried everything except actually governing. Our latest plan involves blaming the last Labour government from 2010. P.S. - Anyone seen Liz Truss?",
                payload: RichMessage.Part(headers: [:], mimeType: "text/plain", parts: []),
                labels: [Label.primary, Label.important],
                internalDate: Date(timeIntervalSinceNow: -7200),
                extent: .metadata
            )
        ),
        
        // Labour Party
        Message(
            id: "002",
            threadId: "wet-lettuce-leadership",
            user: User.samples.first!,
            rich: RichMessage(
                id: "002",
                threadId: "wet-lettuce-leadership",
                recipients: RichMessage.Recipients(
                    from: Contact(address: "keir.starmer@labour.org.uk"),
                    to: [Contact.samples.first!],
                    cc: [],
                    bcc: []
                ),
                subject: "Exciting New Policy Announcement",
                snippet: "After extensive consultation, we've decided to have an opinion about something. It's very sensible and won't upset anyone. We're also banning fun.",
                payload: RichMessage.Part(headers: [:], mimeType: "text/plain", parts: []),
                labels: [Label.promotions],
                internalDate: Date(timeIntervalSinceNow: -3600),
                extent: .metadata
            )
        ),
        
        // Reform UK / Farage
        Message(
            id: "003",
            threadId: "immigration-obsession",
            user: User.samples.first!,
            rich: RichMessage(
                id: "003",
                threadId: "immigration-obsession",
                recipients: RichMessage.Recipients(
                    from: Contact(address: "nigel.farage@reformuk.com"),
                    to: [Contact.samples.first!],
                    cc: [],
                    bcc: []
                ),
                subject: "URGENT: The Boats Are Coming",
                snippet: "Just spotted a rubber dinghy in Dover. Clearly this is why your nan can't get a doctor's appointment. Also, something about sovereignty. Pint?",
                payload: RichMessage.Part(headers: [:], mimeType: "text/plain", parts: []),
                labels: [Label.primary],
                internalDate: Date(timeIntervalSinceNow: -1800),
                extent: .metadata
            )
        ),
        
        // Green Party
        Message(
            id: "004",
            threadId: "tree-huggers-unite",
            user: User.samples.first!,
            rich: RichMessage(
                id: "004",
                threadId: "tree-huggers-unite",
                recipients: RichMessage.Recipients(
                    from: Contact(address: "caroline.lucas@greenparty.org.uk"),
                    to: [Contact.samples.first!],
                    cc: [],
                    bcc: []
                ),
                subject: "Mortgage Your Children for Climate Action",
                snippet: "We propose replacing all roads with wildflower meadows. Yes, ambulances will struggle, but think of the bees! Also, your heating bills funding our hemp commune.",
                payload: RichMessage.Part(headers: [:], mimeType: "text/plain", parts: []),
                labels: [Label.primary],
                internalDate: Date(timeIntervalSinceNow: -5400),
                extent: .metadata
            )
        ),
        
        // Liberal Democrats
        Message(
            id: "005",
            threadId: "fence-sitting-championship",
            user: User.samples.first!,
            rich: RichMessage(
                id: "005",
                threadId: "fence-sitting-championship",
                recipients: RichMessage.Recipients(
                    from: Contact(address: "ed.davey@libdems.org.uk"),
                    to: [Contact.samples.first!],
                    cc: [],
                    bcc: []
                ),
                subject: "We Exist! Remember Us?",
                snippet: "Our latest bold policy: being quite nice about things. We're neither left nor right, just permanently confused. Coalition anyone?",
                payload: RichMessage.Part(headers: [:], mimeType: "text/plain", parts: []),
                labels: [Label.primary],
                internalDate: Date(timeIntervalSinceNow: -9000),
                extent: .metadata
            )
        ),
        
        // UKIP (Still somehow exists)
        Message(
            id: "009",
            threadId: "remember-us",
            user: User.samples.first!,
            rich: RichMessage(
                id: "009",
                threadId: "remember-us",
                recipients: RichMessage.Recipients(
                    from: Contact(address: "someone@ukip.org"),
                    to: [Contact.samples.first!],
                    cc: [],
                    bcc: []
                ),
                subject: "We're Still Here! Sort Of...",
                snippet: "Brexit is done but we're still angry about something. Probably Europe. Or foreigners. Or foreign Europeans. Please vote for us. Anyone?",
                payload: RichMessage.Part(headers: [:], mimeType: "text/plain", parts: []),
                labels: [Label.primary],
                internalDate: Date(timeIntervalSinceNow: -86400),
                extent: .metadata
            )
        ),
        
        // Bonus: Monster Raving Loony Party
        Message(
            id: "010",
            threadId: "actually-sensible",
            user: User.samples.first!,
            rich: RichMessage(
                id: "010",
                threadId: "actually-sensible",
                recipients: RichMessage.Recipients(
                    from: Contact(address: "leader@omrlp.com"),
                    to: [Contact.samples.first!],
                    cc: [],
                    bcc: []
                ),
                subject: "Our Policies Are Still More Coherent",
                snippet: "We promise to replace the House of Lords with a giant hamster wheel. Somehow this still makes more sense than everyone else's manifestos.",
                payload: RichMessage.Part(headers: [:], mimeType: "text/plain", parts: []),
                labels: [Label.primary],
                internalDate: Date(timeIntervalSinceNow: -43200),
                extent: .metadata
            )
        )
    ]
}

