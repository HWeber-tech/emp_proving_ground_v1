from src.sensory.dimensions.why.yield_signal import YieldSlopeTracker


def test_yield_features_basic():
    yt = YieldSlopeTracker()
    yt.update("2Y", 4.0)
    yt.update("5Y", 4.3)
    yt.update("10Y", 4.5)
    yt.update("30Y", 4.7)
    assert yt.slope_2s10s() == 0.5
    assert abs(yt.slope_5s30s() - 0.4) < 1e-9
    assert yt.curvature_2_10_30() == (2 * 4.5 - 4.0 - 4.7)
    ps = yt.parallel_shift()
    assert ps is not None and abs(ps - ((4.0 + 4.3 + 4.5 + 4.7) / 4.0)) < 1e-9
