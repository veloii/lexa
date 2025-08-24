import Foundation
import SwiftData
internal import UIKit

nonisolated struct ParsedEmailAddress {
    let name: String?
    let address: String
    init(name: String? = nil, address: String) {
        self.name = name
        self.address = address
    }
}

extension RichMessage.Part {
    var allParts: [RichMessage.Part] {
        [self] + parts.flatMap(\.allParts)
    }
    
    func content() -> RichMessage.Part? {
        var textPart: RichMessage.Part?
        
        for part in allParts {
            guard part.body != nil else {
                continue
            }
            guard part.fileName == nil else {
                continue
            }
            
            if part.mimeType == "text/plain" {
                textPart = part
            }
            
            if part.mimeType == "text/html" {
                return part
            }
        }
        
        return textPart
    }
}

extension RichMessage.Part.Body {
    static private func decodeURLSafeBase64(_ encodedDataString: String) -> Data? {
        let cleanedString = encodedDataString
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "-", with: "+")
            .replacingOccurrences(of: "_", with: "/")
        
        let paddingLength = (4 - (cleanedString.count % 4)) % 4
        let paddedString = cleanedString + String(repeating: "=", count: paddingLength)
        
        return Data(base64Encoded: paddedString)
    }
    
    static func parse(from gmailApiBody: GmailApi.MessagePartBody?) -> Self? {
        guard let encodedDataString = gmailApiBody?.data, let size = gmailApiBody?.size, size != 0 else {
            return nil
        }
        
        guard let data = decodeURLSafeBase64(encodedDataString) else {
            return nil
        }
        guard let dataString = String(data: data, encoding: .utf8) else {
            return nil
        }
        
        return self.init(size: size, data: dataString)
    }
}

extension RichMessage.Recipients {
    enum ParsingError: Error {
        case missingHeader(String)
        case emptyHeader(String)
        case malformedAddress(String)
        case missingFromAddress
    }
    
    static func parseEmailAddresses(from string: String) throws -> [ParsedEmailAddress] {
        try string.components(separatedBy: ",").map(parseEmailAddress)
    }
    
    static func parseEmailAddress(from string: String) throws -> ParsedEmailAddress {
        let regexWithBrackets = /^(.+?)?\s*(?:<|\u{003c})([^<>\u{003c}\u{003e}]+)(?:>|\u{003e})$/
        
        if let match = string.firstMatch(of: regexWithBrackets) {
            let rawName = match.1?.trimmingCharacters(in: .whitespaces)
            let email = String(match.2).trimmingCharacters(in: .whitespaces)
            let name = rawName?.isEmpty == true ? nil : rawName
            
            return ParsedEmailAddress(name: name, address: email)
        }
        
        let email = string.trimmingCharacters(in: .whitespaces)
        
        guard email.contains("@") else {
            throw ParsingError.malformedAddress(string)
        }
        
        return ParsedEmailAddress(name: nil, address: email)
    }
    
    static func parseRequiredAddressHeader(from headers: [String: String], on key: String) throws -> ParsedEmailAddress {
        guard let headerValue = headers[key] else {
            throw ParsingError.missingHeader(key)
        }
        guard !headerValue.isEmpty else {
            throw ParsingError.emptyHeader(key)
        }
        return try parseEmailAddress(from: headerValue)
    }
    
    static func parseRequiredAddressesHeader(from headers: [String: String], on key: String) throws -> [ParsedEmailAddress] {
        guard let headerValue = headers[key] else {
            throw ParsingError.missingHeader(key)
        }
        guard !headerValue.isEmpty else {
            throw ParsingError.emptyHeader(key)
        }
        return try parseEmailAddresses(from: headerValue)
    }
    
    static func parseAddressesHeader(from headers: [String: String], on key: String) throws -> [ParsedEmailAddress] {
        guard let headerValue = headers[key] else {
            return []
        }
        guard !headerValue.isEmpty else {
            return []
        }
        return try parseEmailAddresses(from: headerValue)
    }
    
    convenience init(from headers: [String: String], context: ModelContext) throws {
        let fromAddress = try Self.parseRequiredAddressHeader(from: headers, on: "From")
        let toAddresses = try Self.parseRequiredAddressesHeader(from: headers, on: "To")
        let ccAddresses = try Self.parseAddressesHeader(from: headers, on: "Cc")
        let bccAddresses = try Self.parseAddressesHeader(from: headers, on: "Bcc")
        
        let allAddresses = [fromAddress] + toAddresses + ccAddresses + bccAddresses
        let allContacts = try getContacts(fromAddresses: allAddresses, using: context)
        let lookup = Dictionary(uniqueKeysWithValues: allContacts.map { ($0.address, $0) })
        
        guard let from = lookup[fromAddress.address] else {
            throw ParsingError.missingFromAddress
        }
        
        self.init(
            from: from,
            to: toAddresses.compactMap { lookup[$0.address] },
            cc: ccAddresses.compactMap { lookup[$0.address] },
            bcc: bccAddresses.compactMap { lookup[$0.address] }
        )
    }
}

extension RichMessage.Part {
    convenience init(from gmailApiPart: GmailApi.MessagePart) {
        let headers = gmailApiPart.headers?.reduce(into: [String : String](), { partialResult, header in
            if let value = header.value, let name = header.name {
                partialResult.self[name] = value
            }
        }) ?? [:]
        
        let parts = gmailApiPart.parts?.compactMap { RichMessage.Part(from: $0) } ?? []
        
        let fileName: String? = if (gmailApiPart.filename?.isEmpty ?? true) { nil } else {
            gmailApiPart.filename
        }
        
        let partId: String? = if (gmailApiPart.partId?.isEmpty ?? true) { nil } else {
            gmailApiPart.partId
        }
        
        self.init(
            headers: headers,
            fileName: fileName,
            mimeType: gmailApiPart.mimeType ?? "unknown",
            partId: partId,
            body: Body.parse(from: gmailApiPart.body),
            parts: parts
        )
    }
}

extension RichMessage {
    enum ParsingError: Error {
        case missingPayload
        case missingDate
    }
    
    convenience init(from message: GmailApi.Message, extent: Extent, using context: ModelContext) throws {
        let part = if let payload = message.payload {
            Part(from: payload)
        } else { throw ParsingError.missingPayload }
        let recipients = try Recipients(from: part.headers, context: context)
        let labels = try [Label](from: message.labelIds ?? [], using: context)
        let subject = part.headers["Subject"]
        
        guard let internalDateString = message.internalDate, let internalDate = TimeInterval(internalDateString) else {
            throw ParsingError.missingDate
        }
        
        self.init(
            id: message.id,
            threadId: message.threadId,
            recipients: recipients,
            subject: subject,
            snippet: message.snippet,
            payload: part,
            labels: labels,
            historyId: message.historyId,
            internalDate: Date(timeIntervalSince1970: internalDate),
            extent: extent
        )
    }
    
    func update(from message: GmailApi.Message, extent: Extent, using context: ModelContext) throws {
        self.labels = try [Label](from: message.labelIds ?? [], using: context)
        
        if extent == .full {
            self.payload = if let payload = message.payload {
                Part(from: payload)
            } else { throw ParsingError.missingPayload }
        }
    }
}

fileprivate func getContacts(fromAddresses emailAddresses: [ParsedEmailAddress], using context: ModelContext) throws -> [Contact] {
    guard !emailAddresses.isEmpty else { return [] }
    
    let emailAddressSet = Set(emailAddresses.map(\.address))
    
    let fetchDescriptor = FetchDescriptor<Contact>(
        predicate: #Predicate<Contact> {
            emailAddressSet.contains($0.address)
        }
    )
    
    let existingContacts = try context.fetch(fetchDescriptor)
    let existingEmailAddressSet = Set(existingContacts.map(\.address))
    
    let missingEmailAddresses = emailAddressSet.subtracting(existingEmailAddressSet)
    let newContacts = missingEmailAddresses.map { emailAddress in
        let label = Contact(address: emailAddress)
        context.insert(label)
        return label
    }
    
    return existingContacts + newContacts
}

extension Array<Label> {
    init(from labelIDs: [String], using context: ModelContext) throws {
        guard !labelIDs.isEmpty else {
            self.init()
            return
        }
        
        let labelIDSet = Set(labelIDs)
        
        let fetchDescriptor = FetchDescriptor<Label>(
            predicate: #Predicate<Label> { labelIDs.contains($0.id) }
        )
        
        let existingLabels = try context.fetch(fetchDescriptor)
        let existingIDSet = Set(existingLabels.map(\.id))
        
        let missingIDs = labelIDSet.subtracting(existingIDSet)
        let newLabels = missingIDs.map { id in
            let label = Label(id: id)
            context.insert(label)
            return label
        }
        
        self.init(existingLabels + newLabels)
    }
}

extension Message {
    convenience init(from: GmailApi.Message, user: User) throws {
        self.init(id: from.id, threadId: from.threadId, user: user)
    }
}
