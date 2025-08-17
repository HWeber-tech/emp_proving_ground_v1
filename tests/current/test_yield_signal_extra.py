from src.sensory.dimensions.why.yield_signal import YieldSlopeTracker


def test_update_none_tenor_noop():
    yt = YieldSlopeTracker()
    yt.update(None, 3.0)  # should no-op  # type: ignore[arg-type]
    assert None not in yt.latest_values
    assert yt.slope() is None  # still no data


def test_signal_missing_returns_zero():
    yt = YieldSlopeTracker()
    # Only one side present
    yt.update("2Y", 4.0)
    sig, conf = yt.signal()
    assert sig == 0.0 and conf == 0.0


def test_signal_flat_and_confidence_floor():
    yt = YieldSlopeTracker()
    yt.update("2Y", 4.0)
    yt.update("10Y", 4.0)
    sig, conf = yt.signal()
    assert sig == 0.0 and conf == 0.2  # floor per implementation


def test_signal_positive_and_confidence_cap():
    yt = YieldSlopeTracker()
    yt.update("2Y", 4.0)
    yt.update("10Y", 4.3)  # slope = 0.3 => conf = min(1.0, 0.3*5 = 1.5) -> 1.0
    sig, conf = yt.signal()
    assert sig == 1.0
    assert conf == 1.0


def test_extended_slope_helpers_and_curvature_and_shift():
    yt = YieldSlopeTracker()
    yt.update("2Y", 4.0)
    yt.update("5Y", 4.3)
    yt.update("10Y", 4.5)
    yt.update("30Y", 4.7)

    # Overloaded slope() and helpers
    assert yt.slope("2Y", "10Y") == 0.5
    assert yt.slope_5s30s() == 0.4

    # Curvature: 2*10Y - 2Y - 30Y
    assert yt.curvature_2_10_30() == (2.0 * 4.5 - 4.0 - 4.7)

    # Parallel shift: average level
    ps = yt.parallel_shift()
    assert ps is not None
    assert abs(ps - ((4.0 + 4.3 + 4.5 + 4.7) / 4.0)) < 1e-9