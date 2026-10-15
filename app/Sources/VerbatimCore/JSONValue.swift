import Foundation

/// An arbitrary JSON value, used for loosely structured engine data such as the
/// "effective settings" record of a job.
public enum JSONValue: Codable, Hashable, Sendable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else if let value = try? container.decode([String: JSONValue].self) {
            self = .object(value)
        } else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unsupported JSON value")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case .bool(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .string(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        }
    }

    public subscript(key: String) -> JSONValue? {
        if case .object(let object) = self { return object[key] }
        return nil
    }

    /// Looks up a snake_case engine key, also accepting its camelCase form
    /// (JSONDecoder's key strategy may or may not rewrite dictionary keys).
    public func lookup(_ snakeKey: String) -> JSONValue? {
        guard case .object(let object) = self else { return nil }
        return object[snakeKey] ?? object[snakeKey.snakeToCamelCase]
    }

    public var stringValue: String? {
        if case .string(let value) = self { return value }
        return nil
    }

    public var doubleValue: Double? {
        if case .number(let value) = self { return value }
        return nil
    }

    /// Compact human-readable rendering for display.
    public var displayString: String {
        switch self {
        case .null: return "—"
        case .bool(let value): return value ? "yes" : "no"
        case .number(let value):
            return value.rounded() == value && abs(value) < 1e15 ? String(Int64(value)) : String(value)
        case .string(let value): return value
        case .array(let values): return values.map(\.displayString).joined(separator: ", ")
        case .object(let object):
            return object.keys.sorted().map { "\($0): \(object[$0]!.displayString)" }.joined(separator: "; ")
        }
    }
}

extension String {
    var snakeToCamelCase: String {
        let parts = split(separator: "_", omittingEmptySubsequences: false)
        guard parts.count > 1 else { return self }
        return String(parts[0]) + parts.dropFirst().map { $0.prefix(1).uppercased() + $0.dropFirst() }.joined()
    }
}
