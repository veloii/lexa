import Foundation
import SwiftUI
import Yams

extension EnvironmentValues {
    @Entry var lexaApiClient: LexaApiClient? = nil
}

protocol LexaApiClient {
    func perform<R: LexaApi.Request>(_ request: R) async throws -> R.ResponseType
}


// MARK: - Live Request Performer Implementation

class LiveLexaApiClient: LexaApiClient {
    private let baseURL: URL
    
    init(baseURL: URL) {
        self.baseURL = baseURL
    }
    
    func perform<R: LexaApi.Request>(_ request: R) async throws -> R.ResponseType {
        guard var urlComponents = URLComponents(url: baseURL.appendingPathComponent(request.path), resolvingAgainstBaseURL: false) else {
            throw NetworkError.invalidURL
        }
        urlComponents.queryItems = request.queryParameters?.map { URLQueryItem(name: $0.key, value: $0.value) }
        
        guard let url = urlComponents.url else {
            throw NetworkError.invalidURL
        }
        
        var urlRequest = URLRequest(
            url: url,
            cachePolicy: .reloadIgnoringLocalAndRemoteCacheData
        )
        urlRequest.httpMethod = request.method.rawValue
        
        if let body = request.body {
            urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
            let encoder = JSONEncoder()
            urlRequest.httpBody = try encoder.encode(body)
        }
        
        let (data, response) = try await URLSession.shared.data(for: urlRequest)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw NetworkError.invalidResponse
        }
        
        switch httpResponse.statusCode {
        case 200...299:
            // For Void return types, we don't need to decode
            if R.ResponseType.self == Void.self {
                // Provide a dummy value that satisfies the Decodable requirement for Void.
                // This is a common workaround as Void itself isn't Decodable.
                return UnmanagedVoid() as! R.ResponseType
            }
            
            let decoder = request.response.decoder()
            return try decoder.decode(R.ResponseType.self, from: data)
        case 401:
            throw NetworkError.unauthenticated
        case 400...499:
            throw NetworkError.clientError(statusCode: httpResponse.statusCode, responseData: data)
        case 500...599:
            throw NetworkError.serverError(statusCode: httpResponse.statusCode, responseData: data)
        default:
            throw NetworkError.unexpectedStatusCode(statusCode: httpResponse.statusCode, responseData: data)
        }
    }
}

nonisolated struct JSONLDecoder: ApiDecoder {
    func decode<D: Decodable>(_ type: D.Type, from data: Data) throws -> D {
        guard let jsonlString = String(data: data, encoding: .utf8) else {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(codingPath: [], debugDescription: "Invalid UTF-8 data")
            )
        }
        
        let lines = jsonlString.components(separatedBy: .newlines)
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        
        guard lines.count > 1 else {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(codingPath: [], debugDescription: "Not valid JSON or JSONL")
            )
        }
        
        let jsonArrayString = "[\(lines.joined(separator: ","))]"
        guard let jsonData = jsonArrayString.data(using: .utf8) else {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(codingPath: [], debugDescription: "Failed to convert JSONL to JSON")
            )
        }
        
        return try JSONDecoder().decode(type, from: jsonData)
    }
}

enum JSONLDecoderError: Error {
    case failedToConvertToStringData
    case typeNotArray
    case emptyStringData
}

extension LexaApi.HTTPResponseType {
    func decoder() -> any ApiDecoder {
        switch self {
        case .json:
            JSONDecoder()
        case .yaml:
            YAMLDecoder()
        case .jsonl:
            JSONLDecoder()
        }
    }
}
