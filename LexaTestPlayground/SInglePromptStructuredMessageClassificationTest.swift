import SwiftUI
import Charts
import Combine
import FoundationModels

extension EnvironmentValues {
    @Entry var singlePromptTestConfig: SinglePromptStructuredMessageTestConfig? = nil
}

@Observable
class SinglePromptStructuredMessageTestConfig {
    var prompt: String
    var iterationsPerInput: Int
    
    init(
        prompt: String = "Classify the text input by the user.\n\n[properties]\n\nOutput the FIRST label that most closely describes the content.",
        iterationsPerInput: Int = 1
    ) {
        self.prompt = prompt
        self.iterationsPerInput = iterationsPerInput
    }
    
    func buildPrompt() -> FoundationModelStructuredMessageService.Configuration.Instructions.ClassifierPrompt { {
        let properties = $0.map {
            guard let desc = $0.desc else { return $0.label }
            return "\($0.label): \(desc)"
        }.joined(separator: "\n")
        
        return self.prompt.replacingOccurrences(of: "[properties]", with: properties)
    } }
}

extension Array where Element == SinglePromptStructuredMessageClassificationTest.Response.InputStatus {
    func totalCorrect() -> Int {
        self.filter {
            switch $0 {
            case .success(.correct): true
            default: false
            }
        }.count
    }
    
    func accuracy(total: Int) -> Double {
        Double(totalCorrect()) / Double(total)
    }
}

struct SinglePromptStructuredMessageClassificationTest {
    typealias Query = QueryTaskStatus<Response, Response>
    
    struct Response {
        enum Answer {
            case correct
            case incorrect(given: String)
            
            init(
                messages: [LexaApi.LanguageModelMessage],
                comparingTo defNode: StructuredMessageDefNode
            ) throws {
                guard let message = messages.first(role: .assistant) else {
                    throw LexaApi.LanguageModelMessageError.missingRole(.assistant)
                }
                if message.content.lowercased() == defNode.name.lowercased() {
                    self = .correct
                } else {
                    self = .incorrect(given: defNode.name)
                }
            }
        }
        
        typealias InputStatus = Result<Answer, any Error>
        
        var inputStatuses: Dictionary<String, [InputStatus]>
        var totalOperations: Int
        
        init(inputStatuses: Dictionary<String, [InputStatus]> = [:], totalOperations: Int) {
            self.inputStatuses = inputStatuses
            self.totalOperations = totalOperations
        }
        
        func partial(subject: any Subject<Response, Never>) -> PartialResponse {
            PartialResponse(subject: subject, initialResponse: self)
        }
        
        var completedOperations: Int {
            self.inputStatuses.reduce(0, { acc, curr in
                acc + curr.value.count
            })
        }
        
        var progress: Double {
            Double(self.completedOperations) / Double(self.totalOperations)
        }
    }
    
    struct PartialResponse {
        private var subject: any Subject<Response, Never>
        private var data: Response
        
        init(subject: any Subject<Response, Never>, initialResponse data: Response) {
            self.subject = subject
            self.data = data
            
            save()
        }
        
        private func save() {
            self.subject.send(data)
        }
        
        mutating func appendInputStatus(_ data: Response.InputStatus, forKey key: String) {
            if self.data.inputStatuses[key] == nil {
                self.data.inputStatuses[key] = []
            }
            self.data.inputStatuses[key]!.append(data)
            save()
        }
        
        func finish() -> Response { data }
    }
    
    var config: SinglePromptStructuredMessageTestConfig
    var inputRepo: StructuredMessageInputRepo
    var definitionRepo: StructuredMessageDefRepo
    
    var configuration: FoundationModelStructuredMessageService.Configuration { get throws {
        var initial = try FoundationModelStructuredMessageService.Configuration.default
        initial.instructions.classifierPrompt = config.buildPrompt()
        
        let localURL = URL(filePath: "/Users/veloi/Downloads/myrun.fmadapter")
        let adapter = try SystemLanguageModel.Adapter(fileURL: localURL)
        initial.model = .init(
            adapter: adapter,
            guardrails: .permissiveContentTransformations
        )
        
        return initial
    }}
    
    func createService() throws -> FoundationModelStructuredMessageService {
        .init(
            configuration: try self.configuration,
            repo: self.definitionRepo
        )
    }
    
    func perform(subject: any Subject<Response, Never>) async throws -> Response {
        let service = try createService()
        let rootNodes = try definitionRepo.listRootNodes()
        let inputs = inputRepo.values
        let totalOperations = self.config.iterationsPerInput * inputs.count
        
        var partialResponse = Response(totalOperations: totalOperations).partial(subject: subject)
        
        for (index, input) in inputs.enumerated() {
            for _ in 1...config.iterationsPerInput {
                do {
                    guard let message = input.first(role: .user) else {
                        throw LexaApi.LanguageModelMessageError.missingRole(.user)
                    }
                    
                    let nodePath = try await service.promptClassifyToLeaf(
                        from: rootNodes,
                        messageContent: message.content
                    )
                    let leaf = nodePath.last!
                    
                    partialResponse.appendInputStatus(
                        .success(try .init(messages: input, comparingTo: leaf)),
                        forKey: String(index)
                    )
                } catch let error {
                    partialResponse.appendInputStatus(.failure(error), forKey: String(index))
                }
            }
        }
        
        return partialResponse.finish()
    }
}

extension Array where Self.Element == LexaApi.LanguageModelMessage {
    func first(role: LexaApi.LanguageModelMessage.Role) -> LexaApi.LanguageModelMessage? {
        self.first(where: { $0.role == role })
    }
}

struct SinglePromptStructuredMessageClassificationTestView: View {
    @StrictEnvironment(\.inMemoryStructuredMessageDefRepo) var structuredMessageDefRepo
    @StrictEnvironment(\.structuredMessageInputRepo) var structuredMessageInputRepo
    @StrictEnvironment(\.lexaApiClient) var lexaApiClient
    @StrictEnvironment(\.singlePromptTestConfig) var config
    
    @StateTask var status: SinglePromptStructuredMessageClassificationTest.Query?
    @StateTask var lexaInputsStatus: MutationTaskStatus?
    @StateTask var lexaDefsStatus: MutationTaskStatus?

    var body: some View {
        @Bindable var bindableConfig = config
        
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                VStack {
                    if let data = self.status?.data {
                        ScrollView {
                            Chart(
                                Array(data.inputStatuses),
                                id: \.key
                            ) { element in
                                BarMark(
                                    x: .value("Accuracy", element.value.accuracy(total: config.iterationsPerInput)),
                                    y: .value("Prompt", "Prompt \(element.key)")
                                )
                                .foregroundStyle(.blue)
                                .cornerRadius(4)
                            }
                            .animation(.smooth, value: data.completedOperations)
                            .frame(height: CGFloat(data.inputStatuses.count * 50 + 40))
                            .chartXAxis {
                                AxisMarks(values: .stride(by: 0.1)) { value in
                                    AxisValueLabel {
                                        if let accuracy = value.as(Double.self) {
                                            Text(String(format: "%.0f%%", accuracy * 100))
                                                .font(.caption)
                                        }
                                    }
                                    AxisGridLine()
                                    AxisTick()
                                }
                            }
                            .chartYAxis {
                                AxisMarks { value in
                                    AxisValueLabel {
                                        if let promptLabel = value.as(String.self) {
                                            Text(promptLabel)
                                                .font(.caption)
                                        }
                                    }
                                    AxisGridLine()
                                    AxisTick()
                                }
                            }
                            .chartXScale(domain: 0...1.0)
                        }
                        .frame(maxWidth: .infinity)
                    } else {
                        ContentUnavailableView("No data", systemImage: "chart.bar.yaxis", description: Text("Execute a trial to see data"))
                            .frame(maxWidth: .infinity)
                    }
                }
                .frame(maxHeight: .infinity)
                .overlay(alignment: .bottomLeading) {
                    HStack {
                        Text("^[\(structuredMessageDefRepo.all.count) definition](inflect: true)")
                            .contentTransition(.numericText())
                            .animation(
                                .smooth,
                                value: structuredMessageDefRepo.all.count
                            )
                        Text("^[\(structuredMessageInputRepo.values.count) input](inflect: true)")
                            .contentTransition(.numericText())
                            .animation(.smooth, value: structuredMessageInputRepo.values.count)
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.vertical, 8)
                    .padding(.horizontal, 10)
                    .glassEffect(.regular)
                }
                .overlay(alignment: .bottomTrailing) {
                    if case let .pending(response) = self.status {
                        HStack {
                            Text("\(Int(response.progress * 100))%")
                                .bold()
                                .foregroundColor(.blue)
                                .contentTransition(.numericText())
                                .frame(minWidth: 25)
                            
                            ProgressView(value: response.progress, total: 1.0)
                                .progressViewStyle(.linear)
                                .frame(maxWidth: 150)
                            
                            Text("\(response.completedOperations)/\(response.totalOperations) operations")
                                .contentTransition(.numericText())
                                .frame(minWidth: 90)
                        }
                        .animation(.smooth, value: response.completedOperations)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.vertical, 8)
                        .padding(.horizontal, 10)
                        .glassEffect(.regular)
                    } else {
                        EmptyView()
                    }
                }
                .padding(.all, 8)

                ScrollView {
                    VStack(alignment: .leading, spacing: 24) {
                        VStack(alignment: .leading) {
                            AsyncButton("Start", action: runPromptAnalysis)
                                .buttonStyle(.glassProminent)
                                .buttonSizing(.flexible)
                                .controlSize(.extraLarge)
                        }
                        
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Prompt")
                            TextField(
                                "Prompt",
                                text: $bindableConfig.prompt,
                                axis: .vertical
                            )
                        }

                        VStack(alignment: .leading, spacing: 6) {
                            Text("Iterations per prompt")
                            Stepper(
                                String(config.iterationsPerInput),
                                value: $bindableConfig.iterationsPerInput,
                                in: 1...50
                            )
                            .disabled(self.status?.isPending ?? false)
                            .padding(.leading, 8)
                            .background(.quinary)
                            .clipShape(.rect(cornerRadius: 6))
                        }
                        
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Requirements")
                            HStack {
                                AsyncButton("Retrieve inputs", action: getInputsFromLexa)
                                AsyncButton("Retrieve definitions", action: getDefsFromLexa)
                            }
                            .controlSize(.small)
                            .buttonSizing(.flexible)
                        }
                    }
                    .padding()
                }
                .frame(maxWidth: 300)
                .glassEffect(.regular, in: .rect(cornerRadius: 24))
                .padding(.all, 8)
            }
            .frame(maxHeight: .infinity)
            .alert(error: $lexaInputsStatus.error)
            .alert(error: $lexaDefsStatus.error)
            .alert(error: $status.error)
        }
    }
    
    private func runPromptAnalysis() async {
        let test = SinglePromptStructuredMessageClassificationTest(
            config: config,
            inputRepo: structuredMessageInputRepo,
            definitionRepo: structuredMessageDefRepo
        )
        await $status.resolve {
            try await test.perform(subject: $0)
        }
    }
    
    private func getInputsFromLexa() async {
        await $lexaInputsStatus.resolve {
            structuredMessageInputRepo.values = try await lexaApiClient.perform(LexaApi.GetStructuredMessageTests())
        }
    }
    
    private func getDefsFromLexa() async {
        await $lexaDefsStatus.resolve {
            let root = try await lexaApiClient.perform(LexaApi.GetStructuredMessageDefsRoot())
            print(root)
            var allMessages = [StructuredMessageDefNode]()
            for path in root.absolutePaths {
                do {
                    let messages = try await lexaApiClient.perform(LexaApi.GetStructuredMessageDef(path: path))
                    print(messages)
                    allMessages.append(contentsOf: messages)
                } catch let error {
                    print(error)
                }
            }
            structuredMessageDefRepo.all = allMessages
        }
    }
}

#Preview(traits: .sampleData) {
    NavigationStack {
        SinglePromptStructuredMessageClassificationTestView()
    }
}
