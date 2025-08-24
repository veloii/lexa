import Foundation
import SwiftData
import SwiftSoup
import FoundationModels
import SwiftUI

public protocol StructuredMessageService {
    func availability() -> StructuredMessageServiceAvailability
//    func insertStructure(on message: RichMessage) async throws
}

public enum StructuredMessageServiceAvailability {
    public enum UnavailableReason: Equatable, Sendable {
        /// The device does not support Apple Intelligence.
        case deviceNotEligible
        
        /// Apple Intelligence is not enabled on the system.
        case appleIntelligenceNotEnabled
        
        /// The model(s) aren't available on the user's device.
        ///
        /// Models are downloaded automatically based on factors
        /// like network status, battery level, and system load.
        case modelNotReady
        
        case unknown
        
        func description() -> String {
            switch self {
            case .deviceNotEligible:
                return "Device not eligible"
            case .appleIntelligenceNotEnabled:
                return "Apple Intelligence not enabled"
            case .modelNotReady:
                return "Model not ready"
            case .unknown:
                return "Unknown"
            }
        }
    }
    
    /// The system is ready for making requests.
    case available
    
    /// Indicates that the system is not ready for requests.
    case unavailable(UnavailableReason)
    
    init(from: SystemLanguageModel.Availability) {
        switch from {
        case .available:
            self = .available
        case .unavailable(.appleIntelligenceNotEnabled):
            self = .unavailable(.appleIntelligenceNotEnabled)
        case .unavailable(.deviceNotEligible):
            self = .unavailable(.deviceNotEligible)
        case .unavailable(.modelNotReady):
            self = .unavailable(.modelNotReady)
        @unknown default:
            self = .unavailable(.unknown)
        }
    }
    
    public func bool() -> Bool {
        switch self {
        case .available:
            true
        case .unavailable(_):
            false
        }
    }
}

protocol ClassifierOptionable {
    var option: ClassifierOption { get }
}

struct ClassifierOption: Hashable {
    var label: String
    var desc: String?
}

extension StructuredMessageDefNode: ClassifierOptionable {
    var option: ClassifierOption {
        .init(label: self.name, desc: self.desc)
    }
}

struct StructuredDefNodePath {
    var node: StructuredMessageDefNode
    var path: [StructuredMessageDefNode]
}

extension StructuredDefNodePath: ClassifierOptionable {
    var option: ClassifierOption {
        self.node.option
    }
}

extension FoundationModelStructuredMessageService {
    public struct Configuration {
        public struct Instructions {
            typealias ClassifierPrompt = ([ClassifierOption]) -> String
            var classifierPrompt: ClassifierPrompt 
            var variablePrompt: String
        }
        
        var model: SystemLanguageModel
        var messageContent: LanguageModelMessageContent.Configuration
        var instructions: Instructions
        var temperature: Double
        
        static public var `default`: Configuration {
            get throws {
                .init(
                    model: .init(
                        useCase: .contentTagging,
                        guardrails: .permissiveContentTransformations
                    ),
                    messageContent: .init(
                        htmlParsingWhitelist: try .none().addTags("a").addAttributes("a", "href"),
                        shouldDecodeHTMLEntities: true
                    ),
                    instructions: .init(
                        classifierPrompt: { _ in
                        """
                        Message Classifier
                        You are a classifier. Your job is to read a message and select which category it belongs to.
                        
                        Instructions:
                        - Read the input message
                        - Match it to the category that best fits
                        - Use the examples in parentheses as guides
                        - Choose the most specific match
                        """
                        },
                        variablePrompt: """
                        Extract the properties from the message.
                        
                        Links:
                        - Each text option represents a link
                        - Select the text that best matches the description
                        """
                    ),
                    temperature: 0
                )
            }
        }
    }
}

final public class FoundationModelStructuredMessageService: StructuredMessageService {
    var configuration: Configuration
    private var repo: StructuredMessageDefRepo
    
    public init(configuration: Configuration, repo: StructuredMessageDefRepo) {
        self.configuration = configuration
        self.repo = repo
    }
    
    public func availability() -> StructuredMessageServiceAvailability {
        .init(from: configuration.model.availability)
    }
    
    private func createLanguageModelSession(instructions: String? = nil) -> LanguageModelSession {
        LanguageModelSession(
            model: configuration.model,
            instructions: instructions
        )
    }
    
    func sortClassifiableNode(_ a: StructuredDefNodePath, _ b: StructuredDefNodePath) -> Bool {
        sortNode(a.node, b.node)
    }
    
    func sortNode(_ a: StructuredMessageDefNode, _ b: StructuredMessageDefNode) -> Bool {
        a.index ?? 0 < b.index ?? 0
    }
    
    func classifiableNodes(_ nodes: [StructuredMessageDefNode], path: [StructuredMessageDefNode] = []) -> [StructuredDefNodePath] {
        nodes.flatMap { child in
            child.type == .group
            ? classifiableNodes(child.nodes, path: path + [child])
            : [.init(node: child, path: path + [child])]
        }
    }
    
    func promptClassification<Value: ClassifierOptionable>(nodes: [Value], messageContent: String) async throws -> Value {
        if nodes.count == 1 {
            return nodes.first!
        }
        
        let generationSchema = try GenerationSchema(
            root: DynamicGenerationSchema(
                name: "Type of message",
                anyOf: nodes.map(\.option.label)
            ),
            dependencies: []
        )
        
        let options = GenerationOptions(temperature: self.configuration.temperature)
        let response = try await createLanguageModelSession(
            instructions: self.configuration.instructions
                .classifierPrompt(nodes.map(\.option))
        ).respond(
            to: messageContent,
            schema: generationSchema,
            includeSchemaInPrompt: false,
            options: options
        )
        
        let chosenEmailType = try response.content.value(String.self)
        guard let resultChosen = nodes.first(where: {$0.option.label == chosenEmailType}) else {
            throw FoundationModelStructuredMessageServiceError.nonExistingNodeClassified
        }
        return resultChosen
    }

    func promptClassifyToLeaf(from roots: [StructuredMessageDefNode], messageContent: String) async throws -> [StructuredMessageDefNode] {
        var path: [StructuredMessageDefNode] = []
        var nextNodes = roots
        
        while path.last?.type != .item {
            let anyOf = classifiableNodes(nextNodes).sorted(by: sortClassifiableNode)
            let resultChosen = try await promptClassification(nodes: anyOf, messageContent: messageContent)
            path.append(contentsOf: resultChosen.path)
            
            switch resultChosen.node.type {
            case .category:
                nextNodes = resultChosen.node.nodes
            case .item:
                path.append(resultChosen.node)
            case .group:
                throw FoundationModelStructuredMessageServiceError.filteredGroupNodeReached
            }
        }

        return path
    }
    
    func substituteValues(in dictionary: [String: AnyCodableValue], replacingPrefix prefix: String = "$", with substitution: (String) throws -> AnyCodableValue) rethrows -> [String: AnyCodableValue] {
        var result: [String: AnyCodableValue] = [:]
        
        for (key, value) in dictionary {
            result[key] = try substituteValue(value, replacingPrefix: prefix, with: substitution)
        }
        
        return result
    }
    
    // TODO: create a protocol which will have a way to replace a datatype
    // This is because with optionals we want it to cascade to the object above which is just too complicated for this system
    // For example a button with an optional link should be gone completely. Not just an optional field on the link
    private func substituteValue(_ value: AnyCodableValue, replacingPrefix prefix: String, with substitution: (String) throws -> AnyCodableValue) rethrows -> AnyCodableValue {
        switch value {
        case let .string(stringValue):
            if stringValue.hasPrefix(prefix) {
                return try substitution(stringValue)
            }
            return .string(stringValue)
            
        case let .dictionary(dictValue):
            return .dictionary(try substituteValues(in: dictValue, replacingPrefix: prefix, with: substitution))
            
        case let .array(arrayValue):
            return .array(try arrayValue.map { try substituteValue($0, replacingPrefix: prefix, with: substitution) })
            
        default:
            return value
        }
    }
    
    func promptVariables(path: [StructuredMessageDefNode], messageContent: LanguageModelMessageContent) async throws -> [String : AnyCodableValue] {
        let leaf = path.last!
        
        let schema = path.reduce(into: StructuredMessageDefNode.Schema()) { partialResult, node in
            partialResult.deepMerge(with: node.schema)
        }
        
        let placeholderLinks = messageContent.getPlaceholderLinks()
        
        let generationProperties = schema.map { (key, field) in
            DynamicGenerationSchema.Property(
                name: key,
                description: field.descriptionWithExamples,
                schema: field.type.createSchema(urls: placeholderLinks),
                isOptional: field.optional
            )
        }
        
        let generationSchema = try GenerationSchema(
            root: DynamicGenerationSchema(
                name: leaf.name,
                description: leaf.desc,
                properties: generationProperties
            ),
            dependencies: []
        )
        
        let session = createLanguageModelSession(
            instructions: self.configuration.instructions.variablePrompt
        )
        
        let response = try await session.respond(
            to: messageContent.input,
            schema: generationSchema
        )
        
        return try schema.extractValues(
            from: response.content,
            patchedContent: messageContent.patchedContent
        )
    }
    
    func reduceProperties(path: [StructuredMessageDefNode], variables: [String: AnyCodableValue]) async throws -> StructuredMessageDefNode.Properties {
        var seenVariables = Set<String>()
        
        return try path.reduce(into: StructuredMessageDefNode.Properties()) { partialResult, node in
            seenVariables.formUnion(node.schema.keys)
            
            let result = try substituteValues(in: node.properties) { placeholder in
                let key = String(placeholder.dropFirst())
                guard let variable = variables[key] else {
                    throw FoundationModelStructuredMessageServiceError.undefinedVariable(key)
                }
                if !seenVariables.contains(key) {
                    throw FoundationModelStructuredMessageServiceError.variableUsedBeforeAssignment(key)
                }
                return variable
            }
            
            partialResult.deepMerge(with: result)
        }
    }
    
    func decodeProperties<Value: Decodable>(
        _ properties: StructuredMessageDefNode.Properties,
        to: Value.Type = Value.self
    ) throws -> Value {
        let encoder = JSONEncoder()
        let data = try encoder.encode(properties)
        let decoder = JSONDecoder()
        return try decoder.decode(Value.self, from: data)
    }
    
//    func createMessageCard(payload: RichMessage.Part) async throws -> StructuredMessageCard {
//        let messageContent = try LanguageModelMessageContent(
//            from: payload,
//            with: self.configuration.messageContent
//        )
//
//        let rootNodes = try getRootDefNodes()
//        let nodePath = try await promptClassifyToLeaf(from: rootNodes, messageContent: messageContent.input)
//        let variables = try await promptVariables(path: nodePath, messageContent: messageContent)
//        let properties = try await reduceProperties(path: nodePath, variables: variables)
//
//        let messageCard = try decodeProperties(properties, to: StructuredMessageCard.self)
//        return messageCard
//    }
}

public enum FoundationModelStructuredMessageServiceError: Error {
    case nonExistingNodeClassified
    case filteredGroupNodeReached
    case undefinedVariable(String)
    case variableUsedBeforeAssignment(String)
}

extension StructuredMessageDefNode.Schema {
     func extractValues(from generatedContent: GeneratedContent, patchedContent: LanguageModelMessageContent.PatchedContent) throws -> [String: AnyCodableValue] {
        var extractedValues: [String: AnyCodableValue] = [:]
        
        for (key, field) in self {
            let value = try extractValue(
                from: generatedContent,
                property: key,
                fieldType: field.type,
                patchedContent: patchedContent
            )
            extractedValues[key] = value
        }
        
        return extractedValues
    }
    
     private func extractValue(
        from generatedContent: GeneratedContent,
        property: String,
        fieldType: StructuredMessageDefNode.SchemaField.ModelType,
        patchedContent: LanguageModelMessageContent.PatchedContent
    ) throws -> AnyCodableValue {
        switch fieldType {
        case .string:
            return .string(try generatedContent.value(String.self, forProperty: property))
        case .url:
            let url = try generatedContent.value(String.self, forProperty: property)
            return .string(try patchedContent.getReal(url: url))
        }
    }
}
