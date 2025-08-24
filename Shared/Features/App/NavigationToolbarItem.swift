import SwiftUI
//import AutoResizingSheet

struct NavigationToolbarItem: ToolbarContent {
    @State private var selectionPresented = false
    
    var body: some ToolbarContent {
        ToolbarItem(placement: .navigation) {
            Menu {
                Button("Your Inbox", systemImage: "tray.fill") {
                }
                Button("Switch account", systemImage: "person.crop.circle") {
                    selectionPresented = true
                }
                Button("Settings", systemImage: "gear") {
                }
            } label: {
                Image(systemName: "line.3.horizontal")
            }
            .sheet(isPresented: $selectionPresented) {
                SessionsView()
                    .padding(.horizontal, 16)
                    .padding(.bottom, -20)
                    .padding(.top, 28)
            }
        }
    }
}
