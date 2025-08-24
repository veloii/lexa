import SwiftData
import FoundationModels
import Foundation

@Model
final public class StructuredMessageDefNode: Decodable {
    public struct SchemaField: Codable {
        enum CodingKeys: CodingKey {
            case type, description, constraints, examples, optional
        }
        
        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            type = try container.decode(ModelType.self, forKey: .type)
            desc = try container.decode(String.self, forKey: .description)
            constraints = try container.decodeIfPresent(Constraints.self, forKey: .constraints)
            examples = try container.decodeIfPresent([String].self, forKey: .examples) ?? []
            optional = try container.decodeIfPresent(Bool.self, forKey: .optional) ?? false
        }
        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(type, forKey: .type)
            try container.encode(desc, forKey: .description)
            try container.encode(constraints, forKey: .constraints)
            try container.encode(examples, forKey: .examples)
        }

        public enum ModelType: String, Codable {
            case string
            case url
            
            func createSchema(urls: [String]) -> DynamicGenerationSchema {
                switch self {
                case .string:
                    return DynamicGenerationSchema(type: String.self)
                case .url:
                    return DynamicGenerationSchema(
                        type: String.self,
                        guides: [.anyOf(urls)]
                    )
                }
            }
        }
        
        public struct Constraints: Codable {
            enum CodingKeys: CodingKey {
                case unique
            }
            
            public init(from decoder: Decoder) throws {
                let container = try decoder.container(keyedBy: CodingKeys.self)
                unique = try container.decode(Unique.self, forKey: .unique)
            }
            
            public struct Unique: Codable {
                enum CodingKeys: CodingKey {
                    case on
                }
                
                public init(from decoder: Decoder) throws {
                    let container = try decoder.container(keyedBy: CodingKeys.self)
                    on = try container.decode(On.self, forKey: .on)
                }
                
                public func encode(to encoder: Encoder) throws {
                    var container = encoder.container(keyedBy: CodingKeys.self)
                    try container.encode(on, forKey: .on)
                }

                public enum On: String, Codable {
                    case replace
                }
                
                public var on: On
                
                public init(on: On) {
                    self.on = on
                }
            }
            
            public var unique: Unique
            
            public init(unique: Unique) {
                self.unique = unique
            }
        }
        
        public var type: ModelType
        public var optional: Bool
        public var desc: String
        public var examples: [String]
        public var constraints: Constraints?
        
        public var descriptionWithExamples: String {
            if examples.isEmpty {return desc}
            return "\(desc)\n\nExamples:\(examples.map{"- \($0)"}.joined(separator: "\n"))"
        }
        
        public init(
            type: ModelType,
            optional: Bool,
            desc: String,
            examples: [String],
            constraints: Constraints? = nil
        ) {
            self.type = type
            self.optional = optional
            self.examples = examples
            self.desc = desc
            self.constraints = constraints
        }
    }
    
     public enum ModelType: String, Codable {
        case item
        case group
        case category
    }
    
    public typealias Schema = [String : SchemaField]
    public typealias Properties = [String : AnyCodableValue]

    public var name: String
    public var desc: String?
    public var examples: [String]
    public var index: Int?
    
    public var type: ModelType
    @Relationship(deleteRule: .cascade, inverse: \StructuredMessageDefNode.parent) public var nodes: [StructuredMessageDefNode]
    public var parent: StructuredMessageDefNode?
    
    // We're encoding to data because SwiftData will crash storing AnyCodableValue inside of a Dictionary
    @Transient private var cachedProperties: Properties?
    private var propertiesData: Data
    public var properties: Properties {
        get {
            if let cached = cachedProperties {
                return cached
            }
            let decoded = Self.decode(Properties.self, from: propertiesData, default: [:])
            cachedProperties = decoded
            return decoded
        }
        set {
            cachedProperties = newValue
            propertiesData = Self.encode(newValue)
        }
    }
    
    @Transient private var cachedSchema: Schema?
    private var schemaData: Data
    public var schema: Schema {
        get {
            if let cached = cachedSchema {
                return cached
            }
            let decoded = Self.decode(Schema.self, from: schemaData, default: [:])
            cachedSchema = decoded
            return decoded
        }
        set {
            cachedSchema = newValue
            schemaData = Self.encode(newValue)
        }
    }

    public init(
        name: String,
        desc: String? = nil,
        index: Int? = nil,
        examples: [String] = [],
        schema: Schema,
        properties: Properties,
        type: ModelType,
        nodes: [StructuredMessageDefNode] = [],
        parent: StructuredMessageDefNode? = nil
    ) {
        self.name = name
        self.desc = desc
        self.index = index
        self.examples = examples
        self.schemaData = Self.encode(schema)
        self.cachedSchema = schema
        self.propertiesData = Self.encode(properties)
        self.cachedProperties = properties
        self.type = type
        self.nodes = nodes
        self.parent = parent
    }

    enum CodingKeys: CodingKey {
        case name, description, schema, properties, type, children, examples, index
    }
    
    private static func encode<T: Codable>(_ value: T) -> Data {
        return (try? JSONEncoder().encode(value)) ?? Data()
    }
    
    private static func decode<T: Codable>(_ type: T.Type, from data: Data, default defaultValue: T) -> T {
        return (try? JSONDecoder().decode(type, from: data)) ?? defaultValue
    }
    
    public required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)
        desc = try container.decodeIfPresent(String.self, forKey: .description)
        index = try container.decodeIfPresent(Int.self, forKey: .index)

        let properties = try container.decodeIfPresent(Properties.self, forKey: .properties) ?? [:]
        propertiesData = Self.encode(properties)
        cachedProperties = properties
        
        let schema = try container.decodeIfPresent(Schema.self, forKey: .schema) ?? [:]
        schemaData = Self.encode(schema)
        cachedSchema = schema
        
        type = try container.decode(ModelType.self, forKey: .type)
        nodes = try container.decodeIfPresent([StructuredMessageDefNode].self, forKey: .children) ?? []
        
        examples = try container.decodeIfPresent([String].self, forKey: .examples) ?? []
    }

}

 struct StructuredMessageCard: Codable {
     struct Progress: Codable {
         enum ModelType: String, Codable {
            case timeline
        }
        
        var type: ModelType
        var steps: [String]
        var current: Int
    }
    
     struct Action: Codable {
         enum Variant: String, Codable {
            case prominent
        }

        var variant: Variant?
        var link: String?
        var label: String?
    }
    
    var title: String
    var subtitle: String?
    var actions: [Action]
    var progress: Progress?
}
