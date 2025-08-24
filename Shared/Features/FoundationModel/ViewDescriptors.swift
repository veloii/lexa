import Foundation
import FoundationModels

// Change password
//
// Title
// Description
// HStack
//  Button - Link
//  Button - Predefined Click Event

// Order update
//
// Title
// Description
// Timeline
//   Icons
//   Current icon

//@Generable(description: "Content to be shown in the view. Prefer predefined cases when applicable.")
//enum ViewOutput {
//    case predefined(PredefinedView)
//    case custom(ViewNode)
//}
//
//@Generable()
//enum PredefinedView {
//    case changePassword(ChangePassword)
//}
//
//@Generable()
//enum ChangePassword {
//    let description: String
//    let output: String
//}

@Generable(description: "A structured card representation of an email's key information and available actions")
public struct AIStructuredMessage {
    @Guide(description: "Concise, action-oriented title summarizing the email's purpose (e.g., 'Password Reset Request', 'Order Shipped', 'Meeting Invitation'). Keep under 30 characters.")
    var title: String
    
    @Guide(description: "Brief summary of key details from the email. Include specific items, amounts, dates, or names when relevant. Limit to 1-2 sentences. Omit if the title is self-explanatory.")
    var body: String?
    
    @Guide(description: "Visual progress indicator for multi-step processes like shipping, approvals, or workflows. Use only for emails tracking status through defined stages.")
    var timeline: Timeline?
    
    @Guide(description: "Action buttons for the user to respond to or interact with the email content. Order by importance: primary action first. Maximum 3 buttons.")
    var buttons: [ButtonNode]?
}

@Generable(description: "A visual progress indicator showing the current status within a multi-step process")
public struct Timeline {
    @Guide(description: "Ordered sequence of SF Symbol names representing process stages (e.g., 'cart', 'shippingbox', 'checkmark'). Use 2-5 icons.")
    var icons: [String]
    
    @Guide(description: "Zero-based index of the current/active stage in the process. Must be between 0 and icons.count - 1.")
    var currentIndex: Int
}

@Generable(description: "A tappable control with a short label, a visual style, and an action to perform when tapped")
public struct ButtonNode {
    @Guide(description: "Short label shown on the button (1-3 words, no emojis). Use action verbs when possible (e.g., 'Confirm', 'Track Order', 'View Details').")
    var content: String
    
    @Guide(description: "Visual prominence for the button. Use 'prominent' for primary/recommended actions, 'default' for secondary options.")
    var variant: ButtonVariant
    
    @Guide(description: "The action invoked when the button is tapped. Extract URLs from email links or use 'dismiss' for rejection/cancellation actions.")
    var action: ButtonAction
}

@Generable(description: "The action a button performs: open a URL or dismiss the card")
public enum ButtonAction {
    case url(String)  // Full URL extracted from the email's links
    case dismiss      // Closes the card without further action
}

@Generable(description: "Visual style for a button. Use 'prominent' for primary actions and 'default' for secondary")
public enum ButtonVariant {
    case prominent   // Primary action the sender wants the user to take
    case `default`   // Secondary or alternative actions
}

//@Generable(description: "A UI view tree composed of common layout and content primitives.")
//enum ViewNode {
//    case vstack(StackNode)
////    case hstack(StackNode)
//    case text(TextNode)
//    case button(ButtonNode)
////    case spacer(SpacerNode)
//
////    case primitive(PrimitiveViewNode)
////    case component(ComponentViewNode)
//}
//
//@Generable(description: "A tappable control with a short label, a visual style, and an action to perform when tapped.")
//struct ButtonNode {
//    @Guide(description: "Short label shown on the button (1–3 words, no emojis).")
//    var content: String
//    @Guide(description: "Visual prominence for the button.")
//    var variant: ButtonVariant
//    @Guide(description: "The action invoked when the button is tapped.")
//    var action: ButtonAction
//}
//
//@Generable(description: "The action a button performs: open a URL or dismiss the card.")
//enum ButtonAction {
//    case url(String)
//    case dismiss
//}
//
//@Generable(description: "Visual style for a button. Use `prominent` for primary actions and `default` for secondary.")
//enum ButtonVariant {
//    case prominent
//    case `default`
//}
//
//@Generable(description: "Reusable higher-level views.")
//enum ComponentViewNode {
//    case timeline(TimelineNode)
//}
//
//@Generable(description: "Icon timeline with a selected index.")
//struct TimelineNode {
//    @Guide(description: "Index into icons.")
//    var currentIndex: Int
//    
//    @Guide(description: "SF Symbols names.")
//    var icons: [String]
//}
//
//@Generable(description: "Layout and content primitives.")
//enum PrimitiveViewNode {
//    case vstack(StackNode)
//    case hstack(StackNode)
//    case text(TextNode)
//    case spacer(SpacerNode)
//}
//
//@Generable(description: "A horizontal alignment for stacks.")
//enum HAlign {
//    case leading, center, trailing
//}
//
//@Generable(description: "A vertical alignment for stacks.")
//enum VAlign {
//    case top, center, bottom
//}
//
//@Generable(description: "A stack container with children views.")
//struct StackNode {
//    @Guide(description: "Spacing in points.", .range(0...40))
//    var spacing: Double?
//    
//    var horizontalAlignment: HAlign?
//    var verticalAlignment: VAlign?
//    
//    @Guide(description: "Rendered in order.")
//    var children: [ViewNode]
//}
//
//@Generable(description: "A text label.")
//struct TextNode {
//    @Guide(description: "User-facing text.")
//    var content: String
//    
//    @Guide(description: "Semantic style.")
//    var hierarchy: FontHierarchyElement
//}
//
//@Generable(description: "Semantic text role.")
//enum FontHierarchyElement {
//    case title, body, subheadline
//}
//
//@Generable(description: "Flexible space that pushes content apart.")
//struct SpacerNode {
//    @Guide(description: "Minimum length in points.", .range(0...200))
//    var minLength: Double?
//}
//
////@Generable(description: "SF Symbol image by system name.")
////struct ImageSystemNode {
////    // Constrain the model to system symbol names only.
////    var systemName: String
////    var color: ColorSpec?
////    @Guide(description: "Symbol size in points.", .range(8...120))
////    var size: Double?
////}
