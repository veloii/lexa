import Combine
import SwiftUI
import Foundation
import GoogleSignIn
import KeychainAccess
import SwiftData
#if os(iOS)
internal import UIKit
#elseif os(macOS)
import AppKit
#endif

struct TokenResponse {
    let accessToken: String
    let refreshToken: String
    let expirationDate: Date
}

extension EnvironmentValues {
    @Entry var authenticationService: (any AuthenticationService)? = nil
}

protocol AuthenticationService: Observable, AnyObject {
    var activeUser: User? { get }
    func setup() async throws
    func handleURL(_ url: URL)
    func signIn() async throws
    func switchToSession(_ session: User) async throws
    func removeSession(_ session: User) async throws
    func signOutCurrentSession() async throws
    func signOutAllSessions() async throws
    func forceRefreshCurrentSession() async throws
}

@Observable
final class MockAuthenticationService: AuthenticationService {
    private var activeUserId: String?
    private let modelContext: ModelContext
    
    init(modelContext: ModelContext) {
        self.modelContext = modelContext
    }
    
    private func fetchUsers() throws -> [User] {
        return try modelContext.fetch(FetchDescriptor<User>())
    }
    
    var activeUser: User? {
        guard let activeId = activeUserId else { return nil }
        guard let users = try? fetchUsers() else { return nil }
        return users.first { user in
            user.id == activeId
        }
    }
    
    func setup() async throws {
        await restoreLastActiveSession()
    }
    
    func handleURL(_ url: URL) {
        // no-op
    }
    
    func signIn() async throws {
        let existingUsers = try fetchUsers()
        
        if existingUsers.isEmpty {
            for sampleUser in User.samples {
                let userExists = existingUsers.contains { $0.id == sampleUser.id }
                if !userExists {
                    modelContext.insert(sampleUser)
                }
            }
            try modelContext.save()
        }
        
        let users = try fetchUsers()
        guard let firstUser = users.first else {
            throw MockAuthError.noUsersAvailable
        }
        
        activeUserId = firstUser.id
        print("Mock: Signed in as \(firstUser.name) (\(firstUser.email))")
    }
    
    func switchToSession(_ session: User) async throws {
        guard session.id != activeUserId else { return }
        
        let users = try fetchUsers()
        guard users.contains(where: { $0.id == session.id }) else {
            throw MockAuthError.sessionNotFound
        }
        
        activeUserId = session.id
        print("Mock: Switched to session for \(session.name) (\(session.email))")
    }
    
    func removeSession(_ session: User) async throws {
        try deleteUser(id: session.id)
        try modelContext.save()
        
        if activeUserId == session.id {
            let remainingUsers = try fetchUsers()
            if let nextUser = remainingUsers.first {
                try await switchToSession(nextUser)
            } else {
                activeUserId = nil
            }
        }
        
        print("Mock: Removed session for \(session.name) (\(session.email))")
    }
    
    func signOutCurrentSession() async throws {
        if let activeUser {
            try await removeSession(activeUser)
        }
    }
    
    func signOutAllSessions() async throws {
        let users = try fetchUsers()
        users.forEach(modelContext.delete)
        try modelContext.save()
        
        activeUserId = nil
        print("Mock: Signed out all sessions")
    }
    
    func forceRefreshCurrentSession() async throws {
        guard activeUserId != nil else {
            throw MockAuthError.noActiveSession
        }
        
        try await Task.sleep(nanoseconds: 500_000_000) // 0.5 seconds
        print("Mock: Refreshed current session tokens")
    }
    
    private func restoreLastActiveSession() async {
        if let users = try? fetchUsers(), let firstUser = users.first {
            activeUserId = firstUser.id
            print("Mock: Restored session for \(firstUser.name)")
        }
    }
    
    private func deleteUser(id: String) throws {
        let itemsToDelete = try modelContext.fetch(FetchDescriptor<User>(
            predicate: #Predicate {
                $0.id == id
            }
        ))
        
        for item in itemsToDelete {
            modelContext.delete(item)
        }
    }
}

enum MockAuthError: Error, LocalizedError {
    case noUsersAvailable
    case sessionNotFound
    case noActiveSession
    
    var errorDescription: String? {
        switch self {
        case .noUsersAvailable:
            return "No users available for mock authentication"
        case .sessionNotFound:
            return "The requested session was not found"
        case .noActiveSession:
            return "No active session to refresh"
        }
    }
}

@Observable
final class LiveAuthenticationService: AuthenticationService, TokenProvider {
  private var activeUserId: String?
  private var currentUser: GIDGoogleUser?
  private var refreshingTokensTask: Task<Void, any Error>?
  private let modelContext: ModelContext
    
    var currentToken: String? {
        get async throws {
            if let activeUser {
                try await self.validateAndRefreshTokenIfNeeded(for: activeUser)
            } else {
                nil
            }
        }
    }

    private let keychain = Keychain(service: "io.veloi.google-token")
        .label("Google Tokens")
        .synchronizable(true)
        .accessibility(.afterFirstUnlock)
    
    init(modelContext: ModelContext) {
        self.modelContext = modelContext
    }
    
    private func fetchUsers() throws -> [User] {
        return try modelContext.fetch(FetchDescriptor<User>())
    }
  
    func setup() async throws {
        try setupGoogleSignIn()
        await restoreLastActiveSession()
    }
    
    func handleURL(_ url: URL) {
        GIDSignIn.sharedInstance.handle(url)
    }
  
    var activeUser: User? {
        guard let activeId = activeUserId else { return nil }
        guard let users = try? fetchUsers() else { return nil }
        return users.first { user in
            user.id == activeId 
        }
    }
    
    private func setupGoogleSignIn() throws {
      guard let clientId = Bundle.main.infoDictionary?["GIDClientID"] as? String else {
        throw GoogleSignInError.missingGoogleClientId
      }
      setupConfiguration(clientId: clientId)
    }
    
    private func deleteUser(id: String) throws {
        let itemsToDelete = try modelContext.fetch(FetchDescriptor<User>(
            predicate: #Predicate {
                $0.id == id
            }
        ))
        
        for item in itemsToDelete {
            modelContext.delete(item)
        }
    }
    
    private func setupConfiguration(clientId: String) {
        let config = GIDConfiguration(clientID: clientId)
        GIDSignIn.sharedInstance.configuration = config
    }
    
    private func restoreLastActiveSession() async {
        if let activeUser {
            try? await switchToSession(activeUser)
        }
//        else if GIDSignIn.sharedInstance.hasPreviousSignIn() {
//            Task {
//                let _ = try? await restorePreviousSignIn()
//            }
//        }
    }
    
    func signIn() async throws {
#if os(iOS)
        guard let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
              let window = windowScene.windows.first,
              let rootViewController = window.rootViewController else {
            throw GoogleSignInError.missingViewController
        }
        
        let result = try await signInWithContinuation(presentingViewController: rootViewController)
        try await createNewSession(from: result)
        
#elseif os(macOS)
        guard let window = NSApplication.shared.mainWindow else {
            throw GoogleSignInError.missingWindow
        }
        
        let result = try await signInWithContinuation(presentingWindow: window)
        try await createNewSession(from: result)
#endif
    }
    
#if os(iOS)
    private func signInWithContinuation(presentingViewController: UIViewController) async throws -> GIDSignInResult {
        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<GIDSignInResult, Error>) in
            GIDSignIn.sharedInstance.signIn(
                withPresenting: presentingViewController,
                hint: nil,
                additionalScopes: ["https://mail.google.com/"]
            ) { result, error in
                if let error = error {
                    continuation.resume(throwing: error)
                } else if let result = result {
                    continuation.resume(returning: result)
                } else {
                    continuation.resume(throwing: GoogleSignInError.unknown)
                }
            }
        }
    }
#endif
    
#if os(macOS)
    private func signInWithContinuation(presentingWindow: NSWindow) async throws -> GIDSignInResult {
        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<GIDSignInResult, Error>) in
            GIDSignIn.sharedInstance.signIn(
                withPresenting: presentingWindow,
                hint: nil,
                additionalScopes: ["https://mail.google.com/"]
            ) { result, error in
                if let error = error {
                    continuation.resume(throwing: error)
                } else if let result = result {
                    continuation.resume(returning: result)
                } else {
                    continuation.resume(throwing: GoogleSignInError.unknown)
                }
            }
        }
    }
#endif

    private func createNewSession(from result: GIDSignInResult) async throws {
        try await createNewSession(from: result.user)
    }
    
    private func createNewSession(from user: GIDGoogleUser) async throws {
        let newUser = try User.from(googleUser: user)
        
        try deleteUser(id: newUser.id)
        
        modelContext.insert(newUser)
        try modelContext.save()
        
        activeUserId = newUser.id
        try keychain.set(newUser.id, key: "activeUserId")
    }
    
    func switchToSession(_ session: User) async throws {
        guard session.id != activeUserId else { return }
        
        guard let token = try? session.getToken() else {
            throw GoogleSignInError.sessionExpiration(session.email)
        }
        
        guard let accessToken = token.accessToken,
              let refreshToken = token.refreshToken,
              !accessToken.isEmpty,
              !refreshToken.isEmpty else {
            throw GoogleSignInError.sessionExpiration(session.email)
        }
        
        activeUserId = session.id
        try keychain.set(session.id, key: "activeUserId")
    }
    
    
    // We are doing our own token refreshing now
//    private func restorePreviousSignIn() async throws {
//        let user = try await restorePreviousSignInWithContinuation()
//        currentUser = user
//        
//        let matchingSessions = try modelContext.fetch(FetchDescriptor<User>(
//            predicate: #Predicate {
//                $0.id == user.userID
//            }
//        ))
//
//        if matchingSessions.isEmpty {
//            try await createNewSession(from: user)
//        }
//    }
    
//    private func restorePreviousSignInWithContinuation() async throws -> GIDGoogleUser {
//        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<GIDGoogleUser, Error>) in
//            GIDSignIn.sharedInstance.restorePreviousSignIn { user, error in
//                if let error = error {
//                    continuation.resume(throwing: error)
//                } else if let user = user {
//                    continuation.resume(returning: user)
//                } else {
//                    continuation.resume(throwing: GoogleSignInError.noUser)
//                }
//            }
//        }
//    }
    
    func removeSession(_ session: User) async throws {
        try deleteUser(id: session.id)
        try modelContext.save()
        
        if activeUserId == session.id {
            if let nextSession = try fetchUsers().first {
                try await switchToSession(nextSession)
            } else {
                currentUser = nil
                activeUserId = nil
                GIDSignIn.sharedInstance.signOut()
            }
        }
    }
    
    func signOutCurrentSession() async throws {
        if let activeUser {
            try await removeSession(activeUser)
        }
    }
    
    func signOutAllSessions() async throws {
        try await disconnectWithContinuation()
        
        try fetchUsers().forEach(modelContext.delete)
        try modelContext.save()
        
        activeUserId = nil
        currentUser = nil
        
        try keychain.remove("activeUserId")
    }
    
    private func disconnectWithContinuation() async throws {
        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            GIDSignIn.sharedInstance.disconnect { error in
                if let error = error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: ())
                }
            }
        }
    }
    
    private func validateAndRefreshTokenIfNeeded(for user: User) async throws -> String {
        guard let token = try user.getToken(), let accessToken = token.accessToken else {
            throw GoogleSignInError.noUser
        }
        
        let expirationBuffer: TimeInterval = 300 // 5 minutes
        if let expirationDate = token.tokenExpirationDate,
           Date().addingTimeInterval(expirationBuffer) >= expirationDate {
            
            try await refreshSessionTokens(userId: user.id)
            
            guard let updatedSession = try fetchUsers().first(where: { $0.id == user.id }),
                  let newAccessToken = try updatedSession.getToken()?.accessToken else {
                throw GoogleSignInError.tokenRefreshFailed
            }
            
            return newAccessToken
        }
        
        return accessToken
    }
    
    private func refreshSessionTokens(userId: String) async throws {
      if refreshingTokensTask == nil {
        refreshingTokensTask = Task {
            let users = try fetchUsers()
            guard
                let session = users.first(
                where: { $0.id == userId }),
                let refreshToken = try session.getToken()?.refreshToken,
                !refreshToken.isEmpty else {
              throw GoogleSignInError.tokenRefreshFailed
          }
          
          guard let clientId = getCurrentClientId() else {
              throw GoogleSignInError.tokenRefreshFailed
          }
          
          do {
              let newTokens = try await getNewTokens(
                  refreshToken: refreshToken,
                  clientId: clientId
              )
              
              try updateSessionTokens(userId: userId, token: newTokens)
          } catch {
              if let httpError = error as? GoogleSignInError,
                 case .apiRequestFailed(let statusCode) = httpError,
                 statusCode == 400 {
                throw GoogleSignInError.sessionExpiration(session.email)
              }
            
              throw error
          }
          
          refreshingTokensTask = nil
        }
      }
      
      try await refreshingTokensTask?.value
    }
    
    private func getNewTokens(refreshToken: String, clientId: String) async throws -> TokenResponse {
        let url = URL(string: "https://oauth2.googleapis.com/token")!
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        
        var components = URLComponents()
        components.queryItems = [
            URLQueryItem(name: "client_id", value: clientId),
            URLQueryItem(name: "refresh_token", value: refreshToken),
            URLQueryItem(name: "grant_type", value: "refresh_token")
        ]
        
        guard let bodyData = components.query?.data(using: .utf8) else {
            throw GoogleSignInError.tokenRefreshFailed
        }
        
        request.httpBody = bodyData
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw GoogleSignInError.tokenRefreshFailed
        }
        
        guard httpResponse.statusCode == 200 else {
            throw GoogleSignInError.tokenRequestFailed(data, httpResponse)
        }
        
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let accessToken = json["access_token"] as? String else {
            throw GoogleSignInError.tokenRefreshFailed
        }
        
        let newRefreshToken = json["refresh_token"] as? String
        let expiresIn = json["expires_in"] as? Int ?? 3600
        let expirationDate = Date().addingTimeInterval(TimeInterval(expiresIn))
        
        return TokenResponse(
            accessToken: accessToken,
            refreshToken: newRefreshToken ?? refreshToken,
            expirationDate: expirationDate
        )
    }
    
    private func updateSessionTokens(userId: String, token response: TokenResponse) throws {
        guard let user = try fetchUsers().first(where: { $0.id == userId }) else {
            throw GoogleSignInError.cannotFindUser
        }
        
        try user.setToken(Token.from(response: response))
    }
    
    private func getCurrentClientId() -> String? {
        return Bundle.main.infoDictionary?["GIDClientID"] as? String
    }
  
  func forceRefreshCurrentSession() async throws {
      guard let activeUserId else {
          throw GoogleSignInError.missingActiveSession
      }
  
      try await refreshSessionTokens(userId: activeUserId)
  }
}

enum GoogleSignInError: LocalizedError {
    case unknown
    case noUser
    case missingUserId
    case cannotFindUser
    case tokenRefreshFailed
    case missingActiveSession
    case apiRequestFailed(Int)
    case tokenRequestFailed(Data, HTTPURLResponse)
    case sessionExpiration(String)
    case missingGoogleClientId
    case missingViewController
    
    var errorDescription: String? {
        switch self {
        case .unknown:
            return "Unknown error occurred"
        case .missingUserId:
            return "No user ID found"
        case .cannotFindUser:
            return "No using matching user ID found"
        case .missingActiveSession:
            return "No active session"
        case .missingViewController:
            return "Cannot find presenting view controller"
        case .noUser:
            return "No user found"
        case .tokenRefreshFailed:
            return "Failed to refresh authentication tokens"
        case .missingGoogleClientId:
            return "No Google Client ID found"
        case .apiRequestFailed(let statusCode):
            return "API request failed with status code: \(statusCode)"
        case .sessionExpiration(let id):
          return "Session expired for \(id). Please sign in again."
        case .tokenRequestFailed(let data, let response):
          if let errorString = String(data: data, encoding: .utf8) {
              return "Token refresh errored with status code \(response.statusCode): \(errorString)"
          }
          
          return "Token refresh errored with status code \(response.statusCode)"
        }
    }
}
