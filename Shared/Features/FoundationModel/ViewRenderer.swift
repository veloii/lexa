import SwiftUI
//
//struct GeneratedView: View {
//    let node: ViewNode
//    
//    var body: some View {
//        render(node)
//    }
//    
//    @ViewBuilder
//    private func renderPrimitive(_ node: ViewNode) -> some View {
//        switch node {
//        case .vstack(let s):
//            VStack(
//                alignment: mapHAlign(s.horizontalAlignment),
//                spacing: CGFloat(s.spacing ?? 0)
//            ) {
//                renderChildren(s.children)
//            }
//            
////        case .hstack(let s):
////            HStack(alignment: mapVAlign(s.verticalAlignment), spacing: CGFloat(s.spacing ?? 0)) {
////                renderChildren(s.children)
////            }
//            
//        case .text(let t):
//            let font = mapFont(t.hierarchy)
//            let foregroundStyle = mapForegroundStyle(t.hierarchy)
//            Text(t.content)
//                .font(font)
//                .foregroundStyle(foregroundStyle)
//            
////        case .spacer(let s):
////            Spacer(minLength: CGFloat(s.minLength ?? 0))
//        }
//    }
//    
////    @ViewBuilder
////    private func renderComponent(_ node: ComponentViewNode) -> some View {
////        switch node {
////        case .timeline(let t):
////            Spacer()
////        }
////    }
//
//    @ViewBuilder
//    private func render(_ node: ViewNode) -> some View {
//        renderPrimitive(node)
////        switch node {
////        case .primitive(let p):
////            renderPrimitive(p)
////        case .component(let c):
////            renderComponent(c)
////        }
//    }
//    
//    @ViewBuilder
//    private func renderChildren(_ children: [ViewNode]) -> AnyView {
//        AnyView(ForEach(Array(children.enumerated()), id: \.offset) { _, child in
//            render(child)
//        })
//    }
//    
//    private func mapHAlign(_ a: HAlign?) -> HorizontalAlignment {
//        switch a ?? .center {
//        case .leading: return .leading
//        case .center: return .center
//        case .trailing: return .trailing
//        }
//    }
//    
//    private func mapVAlign(_ a: VAlign?) -> VerticalAlignment {
//        switch a ?? .center {
//        case .top: return .top
//        case .center: return .center
//        case .bottom: return .bottom
//        }
//    }
//    
//    private func mapFont(_ h: FontHierarchyElement) -> Font {
//        switch h {
//        case .body: return .body
//        case .title: return .title2
//        case .subheadline: return .subheadline
//        }
//    }
//    
//    private func mapForegroundStyle(_ h: FontHierarchyElement) -> Color {
//        switch h {
//        case .body: return .primary
//        case .title: return .primary
//        case .subheadline: return .secondary
//        }
//    }
//}
