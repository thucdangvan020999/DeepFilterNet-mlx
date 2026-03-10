import MLX

struct TensorRingBuffer {
    private(set) var values: [MLXArray]
    private(set) var totalWritten: Int = 0

    var capacity: Int { values.count }
    var count: Int { min(totalWritten, capacity) }
    var oldestAbsoluteIndex: Int { max(0, totalWritten - capacity) }

    init(capacity: Int, initial: MLXArray) {
        precondition(capacity > 0, "TensorRingBuffer capacity must be > 0")
        values = Array(repeating: initial, count: capacity)
    }

    mutating func reset() {
        totalWritten = 0
    }

    mutating func push(_ value: MLXArray) {
        values[totalWritten % capacity] = value
        totalWritten += 1
    }

    func get(absoluteIndex: Int) -> MLXArray? {
        guard absoluteIndex >= oldestAbsoluteIndex, absoluteIndex < totalWritten else {
            return nil
        }
        return values[absoluteIndex % capacity]
    }

    func orderedLast(_ n: Int) -> [MLXArray] {
        let k = min(max(0, n), totalWritten)
        guard k > 0 else { return [] }
        let start = totalWritten - k
        return (start..<(start + k)).compactMap { get(absoluteIndex: $0) }
    }
}

struct DeepFilterNetStreamingRings {
    var spec: TensorRingBuffer
    var specLow: TensorRingBuffer
    var encErb: TensorRingBuffer
    var encDf: TensorRingBuffer
    var dfConvp: TensorRingBuffer

    mutating func reset() {
        spec.reset()
        specLow.reset()
        encErb.reset()
        encDf.reset()
        dfConvp.reset()
    }
}

struct DeepFilterNetStreamRecurrentState {
    var encEmb: [MLXArray]?
    var erbDec: [MLXArray]?
    var dfDec: [MLXArray]?

    mutating func reset() {
        encEmb = nil
        erbDec = nil
        dfDec = nil
    }
}
