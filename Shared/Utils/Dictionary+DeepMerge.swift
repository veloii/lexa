extension Dictionary where Key == String, Value == AnyCodableValue {
    mutating nonisolated func deepMerge(with other: [String: AnyCodableValue]) {
        for (key, value) in other {
            if let existingValue = self[key] {
                // Check if both values are dictionaries within AnyCodableValue
                if case .dictionary(let existingDict) = existingValue,
                   case .dictionary(let newDict) = value {
                    // Deep merge the inner dictionaries
                    var mutableExisting = existingDict
                    mutableExisting.deepMerge(with: newDict)
                    self[key] = .dictionary(mutableExisting)
                } else {
                    // Not both dictionaries, replace the value
                    self[key] = value
                }
            } else {
                // Key doesn't exist, add it
                self[key] = value
            }
        }
    }
}

// Generic version (if you need it for other dictionary types)
extension Dictionary {
    mutating nonisolated func deepMerge(with other: [Key: Value]) {
        for (key, value) in other {
            if let existingValue = self[key] {
                // Check if both values are dictionaries themselves
                if let existingDict = existingValue as? [AnyHashable: Any],
                   let newDict = value as? [AnyHashable: Any] {
                    // This is a more generic approach for nested dictionaries
                    var mutableExisting = existingDict
                    mutableExisting.merge(newDict) { (_, new) in new }
                    self[key] = mutableExisting as? Value ?? value
                } else {
                    // Not both dictionaries, replace the value
                    self[key] = value
                }
            } else {
                // Key doesn't exist, add it
                self[key] = value
            }
        }
    }
}
