import SwiftUI
import Combine

protocol PublisherTaskable {
    associatedtype Pending
    static func pending(_ pending: Pending) -> Self
    associatedtype Data
    static func data(_ data: Data) -> Self
    static func error(_ error: any Error) -> Self
    var readonlyError: Error? { get }
}

extension Optional: PublisherTaskable where Wrapped: PublisherTaskable {
    static func pending(_ pending: Wrapped.Pending) -> Self {
        .some(.pending(pending))
    }
    static func data(_ data: Wrapped.Data) -> Self {
        .some(.data(data))
    }
    static func error(_ error: any Error) -> Self {
        .some(.error(error))
    }
    var readonlyError: (any Error)? {
        self?.readonlyError
    }
}

@Observable
@propertyWrapper class StateTask<Value>: Sendable {
    var wrappedValue: Value
    var projectedValue: StateTask { self }
    
    init(wrappedValue: Value) {
        self.wrappedValue = wrappedValue
    }
}

extension StateTask where Value: PublisherTaskable {
    func resolve(_ closure: (any Subject<Value.Pending, Never>) async throws -> Value.Data) async {
        do {
            let value = PassthroughSubject<Value.Pending, Never>()
            var cancellables = Set<AnyCancellable>()
            value.receive(on: DispatchQueue.main)
                .sink(receiveValue: {
                        self.wrappedValue = .pending($0)
                    })
                .store(in: &cancellables)
            let result = try await closure(value)
            cancellables.removeAll()
            self.wrappedValue = .data(result)
        } catch let error {
            self.wrappedValue = .error(error)
        }
    }
}

protocol OptionalProtocol {
    associatedtype Wrapped
    var optional: Wrapped? { get set }
}

extension Optional: OptionalProtocol {
    var optional: Wrapped? {
        get { self }
        set { self = newValue }
    }
}

extension StateTask where Value: OptionalProtocol, Value.Wrapped: PublisherTaskable {
    var error: Binding<(any Error)?> {
        Binding {
            self.wrappedValue.optional.error
        } set: { newValue in
            self.wrappedValue.optional.error = newValue
        }
    }
}

extension StateTask where Value == MutationTaskStatus? {
    var error: Binding<(any Error)?> {
        Binding {
            self.wrappedValue.optional.error
        } set: { newValue in
            self.wrappedValue.optional.error = newValue
        }
    }
    
    func resolve(_ closure: () async throws -> Void) async {
        self.wrappedValue = .loading
        do {
            try await closure()
            self.wrappedValue = nil
        } catch let error {
            self.wrappedValue = .error(error)
        }
    }
}

enum QueryTaskStatus<Data, Pending>: PublisherTaskable {
    case data(Data)
    case pending(Pending)
    case error(Error)
    
    var isPending: Bool {
        if case .pending = self {
            return true
        }
        return false
    }
    
    var readonlyError: (any Error)? {
        switch self {
        case .error(let error): error
        default: nil
        }
    }
}

extension QueryTaskStatus where Data == Pending {
    var data: Data? {
        switch self {
        case .data(let data):
            return data
        case .pending(let data):
            return data
        case .error(_):
            return nil
        }
    }
}

enum MutationTaskStatus: Sendable {
    case loading
    case error(Error)
        
    var isLoading: Bool {
        if case .loading = self {
            return true
        }
        return false
    }
}

extension MutationTaskStatus? {
    var isLoading: Bool {
        if let self, case .loading = self {
            return true
        }
        return false
    }
}

struct ErrorAlert: ViewModifier {
    @Binding var error: Error?
    
    var isPresented: Binding<Bool> {
        Binding(get: {
            error != nil
        }, set: {
            if $0 == false {
                error = nil
            }
        })
    }
    
    func body(content: Content) -> some View {
        content.alert("An error has occurred", isPresented: isPresented) {
            Button("OK", role: .cancel) {}
        } message: {
            if let error {
                Text(error.localizedDescription)
            }
        }
    }
}

extension View {
    func alert(error: Binding<Error?>) -> some View {
        modifier(ErrorAlert(error: error))
    }
}

extension Optional<MutationTaskStatus> {
    var error: Error? {
        get {
            if case .error(let error) = self {
                return error
            }
            return nil
        }
        set {
            if let newValue {
                self = .error(newValue)
            } else {
                self = nil
            }
        }
    }
}

extension Optional where Wrapped: PublisherTaskable {
    var error: Error? {
        get { self?.readonlyError }
        set {
            if let newValue {
                self = .error(newValue)
            } else {
                self = nil
            }
        }
    }
}
