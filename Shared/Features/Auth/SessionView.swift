import SwiftData
import SwiftUI
import GoogleSignIn
import Combine

struct SessionsView: View {
    @Query var users: [User]
    
    @StrictEnvironment(\.authenticationService) private var authenticationService
    @State private var signInStatus: MutationTaskStatus?
    
    var body: some View {
        LazyVStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Accounts")
                    .font(.title)
                    .fontWeight(.bold)
                
                Spacer()
                
                AsyncButton(action: signIn) {
                    Image(systemName: "plus")
                }
                .clipShape(.circle)
                .tint(.blue)
                .buttonStyle(.glassProminent)
            }
            .padding(.bottom, 12)
            .padding(.horizontal, 4)

            ForEach(users) { user in
                SessionRow(user: user)
            }
            
        }
        .alert(error: $signInStatus.error)
    }
    
    private func signIn() async {
        signInStatus = .loading
        signInStatus = await .closure {
            try await authenticationService.signIn()
        }
    }
}

struct SessionRow: View {
    let user: User
    @StrictEnvironment(\.authenticationService) private var authenticationService
    @State private var switchToSessionStatus: MutationTaskStatus?
    @State private var removeSessionStatus: MutationTaskStatus?

    var isActive: Bool {
        authenticationService.activeUser?.id == user.id
    }
    
    var body: some View {
            HStack(spacing: 12) {
                AsyncImage(url: URL(string: user.profileImageURL ?? "")) { image in
                    image
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                } placeholder: {
                    Circle()
                        .fill(Color.blue.opacity(0.3))
                        .overlay(
                            Image(systemName: "person.fill")
                                .foregroundColor(.blue)
                                .font(.system(size: 14))
                        )
                }
                .frame(width: 40, height: 40)
                .clipShape(Circle())
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(user.name)
                        .fontWeight(.medium)
                    Text(user.email)
                        .foregroundStyle(.secondary)
                }
                .font(.subheadline)
                
                Spacer()
                
                if isActive {
                    Image(systemName: "checkmark")
                        .foregroundStyle(.selection)
                        .font(.system(size: 18))
                }
                
                //            AsyncButton(action: removeSession) {
                //                Image(systemName: "xmark")
                //                    .font(.system(size: 16))
                //            }
                //            .tint(.red)
                //            .buttonStyle(.glass)
            }
        
//        .tint(.blue)
//        .if(isActive) {
//            $0.buttonStyle(.glassProminent)
//        }
//        .background(Color(UIColor.tertiarySystemBackground))
            .swipeActions(edge: .trailing) {
                Button(role: .destructive, action: {
                    Task {
                        await removeSession()
                    }
                }) {
                    Text("dete")
//                    Label("Delete", systemImage: "trash")
                }
            }
            .contentShape(.rect)
            .onTapGesture {
                if !isActive {
                    Task {
                        await switchToSession()
                    }
                }
            }
            .padding(.all, 8)
            .padding(.trailing)
            .glassEffect(
                .clear
                    .tint(Color(UIColor.tertiarySystemBackground))
                    .interactive()
            )
//        .tint(isActive ? AnyShapeStyle(.selection) : AnyShapeStyle(.black))
//        .glassEffect(.regular.tint(.blue.opacity(0.1)))
        .alert(error: $switchToSessionStatus.error)
        .alert(error: $removeSessionStatus.error)
    }
    
    private func switchToSession() async {
        switchToSessionStatus = .loading
        switchToSessionStatus = await .closure {
            try await authenticationService.switchToSession(user)
        }
    }
    
    private func removeSession() async {
        removeSessionStatus = .loading
        removeSessionStatus = await .closure {
            try await authenticationService.removeSession(user)
        }
    }
}
