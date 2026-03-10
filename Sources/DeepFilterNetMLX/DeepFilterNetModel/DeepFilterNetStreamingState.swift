import MLX

final class TensorRingBuffer {
    private(set) var values: [MLXArray]
    private(set) var totalWritten: Int = 0

    var capacity: Int { values.count }
    var count: Int { min(totalWritten, capacity) }
    var oldestAbsoluteIndex: Int { max(0, totalWritten - capacity) }

    init(capacity: Int, initial: MLXArray) {
        precondition(capacity > 0, "TensorRingBuffer capacity must be > 0")
        values = Array(repeating: initial, count: capacity)
    }

    @inline(__always)
    func reset() {
        totalWritten = 0
    }

    @inline(__always)
    func push(_ value: MLXArray) {
        values[totalWritten % capacity] = value
        totalWritten += 1
    }

    @inline(__always)
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

final class DeepFilterNetStreamingRings {
    let spec: TensorRingBuffer
    let specLow: TensorRingBuffer
    let encErb: TensorRingBuffer
    let encDf: TensorRingBuffer
    let dfConvp: TensorRingBuffer

    init(
        spec: TensorRingBuffer,
        specLow: TensorRingBuffer,
        encErb: TensorRingBuffer,
        encDf: TensorRingBuffer,
        dfConvp: TensorRingBuffer
    ) {
        self.spec = spec
        self.specLow = specLow
        self.encErb = encErb
        self.encDf = encDf
        self.dfConvp = dfConvp
    }

    @inline(__always)
    func reset() {
        spec.reset()
        specLow.reset()
        encErb.reset()
        encDf.reset()
        dfConvp.reset()
    }
}

final class DeepFilterNetStreamRecurrentState {
    var encEmb: [MLXArray]?
    var erbDec: [MLXArray]?
    var dfDec: [MLXArray]?

    @inline(__always)
    func reset() {
        encEmb = nil
        erbDec = nil
        dfDec = nil
    }
}
