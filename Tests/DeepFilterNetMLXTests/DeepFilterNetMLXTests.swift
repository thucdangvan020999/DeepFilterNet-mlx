import Testing
@testable import DeepFilterNetMLX

@Test("DeepFilterNet config defaults")
func configDefaults() {
    let config = DeepFilterNetConfig()
    #expect(config.sampleRate == 48_000)
    #expect(config.fftSize == 960)
    #expect(config.hopSize == 480)
    #expect(config.nbErb == 32)
    #expect(config.nbDf == 96)
    #expect(config.freqBins == 481)
}

@Test("Performance presets")
func performancePresets() {
    let throughput = DeepFilterNetPerformanceConfig.throughput
    #expect(throughput.enableMetalFusedMaskMultiply)
    #expect(throughput.enableMetalFusedStreamingDeepFilter)

    let safe = DeepFilterNetPerformanceConfig.safe
    #expect(!safe.enableMetalFusedMaskMultiply)
    #expect(!safe.enableMetalFusedStreamingDeepFilter)
}
