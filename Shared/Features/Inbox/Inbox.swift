import SwiftUI
import SwiftData

struct OrderDispatchView: View {
    private var email = "google.com"

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack {
                AsyncImage(url: URL(string: "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Google_%22G%22_logo.svg/768px-Google_%22G%22_logo.svg.png")) { image in
                    image.resizable()
                } placeholder: {
                    ProgressView()
                }
                .frame(width: 20, height: 20)
                .clipShape(.circle)
                
                
                Text(email)
                    .font(.callout)
            }

            VStack(alignment: .leading, spacing: 12) {
                Text("Order dispatched")
                    .font(.title2)
                    .fontWeight(.semibold)
                
                Text("Sony SEL90M28G FE 90mm F2.8 Macro Lens")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                
                Spacer()
                
                GlassEffectContainer {
                    HStack {
                        Image(systemName: "cart.fill")
                            .font(.system(size: 16))
                            .frame(width: 40, height: 40)
                            .foregroundStyle(.white)
                            .glassEffect(.regular.tint(.blue), in: .circle)
                        
                        Spacer()
                            .background(
                                Color.clear.frame(minHeight: 4)
                                    .glassEffect(.regular.tint(.blue))
                            )
                        
                        Image(systemName: "shippingbox.fill")
                            .font(.system(size: 16))
                            .frame(width: 40, height: 40)
                            .foregroundStyle(.white)
                            .glassEffect(.regular.tint(.blue), in: .circle)
                        
                        Spacer()
                            .background(
                                Color.clear.frame(minHeight: 4)
                                    .glassEffect(.regular.tint(Color(UIColor.systemFill)))
                            )
                        
                        Image(systemName: "checkmark")
                            .font(.system(size: 16))
                            .frame(width: 40, height: 40)
                            .glassEffect(.regular.tint(Color(UIColor.systemFill)), in: .circle)
                    }
                }
            }
        }
        .padding()
        .frame(maxHeight: .infinity, alignment: .top)
        .glassEffect(
            .regular.tint(Color(UIColor.secondarySystemGroupedBackground)),
            in: RoundedRectangle(cornerRadius: 32)
        )
    }
}

struct ResetPasswordView: View {
    private var email = "google.com"

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                AsyncImage(url: URL(string: "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Google_%22G%22_logo.svg/768px-Google_%22G%22_logo.svg.png")) { image in
                    image.resizable()
                } placeholder: {
                    ProgressView()
                }
                .frame(width: 20, height: 20)
                .clipShape(.circle)
                
                
                Text(email)
                    .font(.callout)
            }
            
            VStack(alignment: .leading, spacing: 12) {
                Text("Reset password request")
                    .font(.title2)
                    .fontWeight(.semibold)
                
                Text("If you didn't make this request, you can dismiss this message.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                
                Spacer(minLength: 0)
                
                HStack(spacing: 8) {
                    Button(action: {}) {
                        Text("Confirm")
                            .frame(maxWidth: .infinity)
                    }
                    
                    .buttonStyle(.glassProminent)
                    
                    Button(action: {}) {
                        Text("Dismiss")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.glass)
                    
//                    Spacer()
                    
//                    Text("Expires in 59m")
//                        .foregroundStyle(.secondary)
//                        .font(.subheadline)
//                        .padding(.trailing, 4)
                }
            }.frame(maxWidth: .infinity)
        }
        .padding()
        .frame(maxWidth: 350, maxHeight: .infinity, alignment: .top)
        .glassEffect(
            .regular.tint(Color(UIColor.secondarySystemGroupedBackground)),
            in: RoundedRectangle(cornerRadius: 32)
        )
    }
}

struct InboxView: View {
    @StrictEnvironment(\.messageService) private var messageService
    @Query(
        filter: #Predicate<RichMessage> { $0.structured == nil },
        sort: [SortDescriptor(\RichMessage.internalDate, order: .reverse)]
    ) var richMessages: [RichMessage]
    @State private var searchText: String = ""
    
    @State private var showingStructuredTest = false
    
    var body: some View {
        NavigationView {
            List {
//                VStack(alignment: .leading, spacing: 12) {
//                    HStack(spacing: 12) {
//                        AsyncImage(url: URL(string: "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Google_%22G%22_logo.svg/768px-Google_%22G%22_logo.svg.png")) { image in
//                            image.resizable()
//                        } placeholder: {
//                            ProgressView()
//                        }
//                        .frame(width: 28, height: 28)
//                        .clipShape(.circle)
//                        
//                        Text("501829")
//                            .font(.title2)
//                            .fontDesign(.monospaced)
//                            .fontWeight(.medium)
//                        
//                        Spacer()
//                        
//                        Text("Expires in 10m")
//                            .font(.subheadline)
//                            .foregroundStyle(.secondary)
//                    }
//                }
//                .padding(.horizontal, 8)
//                .padding(.vertical, 8)
//                .listRowBackground(
//                    Rectangle()
//                        .fill(.clear)
//                        .glassEffect(
//                            .regular.tint(Color(UIColor.secondarySystemGroupedBackground)),
//                            in: RoundedRectangle(cornerRadius: 16)
//                        )
//                        .padding(.horizontal, 12)
//                        .padding(.vertical, 4)
//                )
//                .listRowSeparator(.hidden)
//                .listRowInsets(.horizontal, 24)
//                .listRowInsets(.vertical, 16)
//                .zIndex(0)
                
                Button("Brexit") {
                    showingStructuredTest = true
                }
                
                ScrollView(.horizontal) {
                    VStack(spacing: 20) {
                        HStack(alignment: .top, spacing: 10) {
                            ResetPasswordView()
                            OrderDispatchView()
                        }
                        .scrollTargetLayout()
                        Rectangle()
                            .fill(Color(UIColor.systemFill))
                            .frame(height: 1)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 64)
                }
                .scrollTargetBehavior(.viewAligned)
                .scrollIndicators(.never)
                .padding(.vertical, -64)
                .listRowBackground(Color.clear)
                .listRowInsets(.horizontal, 0)
                .listRowSeparator(.hidden)
                .zIndex(0)
                
                ForEach(Array(richMessages.enumerated()), id: \.element.id) { index, message in
                    HStack(alignment: .top, spacing: 12) {
                        Text("K")
                            .frame(width: 36, height: 36)
                            .foregroundStyle(Color(UIColor.label))
                            .background(Color(UIColor.systemFill))
                            .clipShape(.circle)
                            .padding(.top, 2)
                        
                        VStack(alignment: .leading, spacing: 4) {
                            HStack(alignment: .center) {
                                Text("\(message.recipients.from.address)")
                                
                                Spacer()
                                
                                Circle()
                                    .fill(.blue)
                                    .frame(width: 8)
                            }
                            .padding(.bottom, 2)
                            
                            if let subject = message.subject {
                                Text("\(subject)")
                                    .font(.subheadline)
                            }
                            if let snippet = message.snippet {
                                Text("\(snippet)")
                                    .lineLimit(2)
                                    .foregroundStyle(Color(UIColor.secondaryLabel))
                                    .font(.subheadline)
                            }
                        }
                        .swipeActions(edge: .trailing) {
                            Button("Delete", systemImage: "trash", role: .destructive, action: {})
                        }
                        .swipeActions(edge: .leading) {
                            Button("Star", systemImage: "star", action: {}).tint(.yellow)
                        }
                        .onTapGesture {
                            print(message.id)
                        }
                    }
                    .listRowBackground(
                        Rectangle()
                            .fill(.clear)
                            .glassEffect(
                                .regular.tint(Color(UIColor.secondarySystemGroupedBackground)),
                                in: RoundedRectangle(cornerRadius: 16)
                            )
                            .padding(.horizontal, 12)
                            .padding(.vertical, 4)
                    )
                    .zIndex(Double(index + 1))
                    .listRowSeparator(.hidden)
                    .listRowInsets(.horizontal, 24)
                    .listRowInsets(.vertical, 16)
                }
            }
            .animation(.smooth, value: richMessages)
            .background(Color(UIColor.systemGroupedBackground))
            .listStyle(.plain)
            .toolbarTitleMenu {
                Button("Primary", systemImage: "star.fill", action: {})
                Button("Promotional", systemImage: "tag.fill", action: {})
                Button("Transactions", systemImage: "cart.fill", action: {})
            }
            .toolbar {
                NavigationToolbarItem()
            }
            .toolbar {
                ToolbarItem(placement: .bottomBar) {
                    Button("Filters", systemImage: "line.3.horizontal.decrease") {
                        Task {
                            do {
                                try await messageService.sync()
                            } catch let error {
                                print(error)
                            }
                        }
                    }
                }
                ToolbarSpacer(.flexible, placement: .bottomBar)
                DefaultToolbarItem(kind: .search, placement: .bottomBar)
                ToolbarSpacer(.flexible, placement: .bottomBar)
                ToolbarItem(placement: .bottomBar) {
                    Button("Compose", systemImage: "square.and.pencil") {}
                }
            }
//            .searchable(
//                text: $searchText,
//                placement: .toolbar
//            )
            .searchToolbarBehavior(.minimize)
            .searchable(
                text: $searchText,
                placement: .toolbar
            )
            .navigationTitle("Primary")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

#Preview(traits: .sampleData) {
    InboxView()
}
