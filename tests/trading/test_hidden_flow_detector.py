from datetime import datetime, timedelta

from src.trading.liquidity import HiddenFlowDetector


def _ts(offset_seconds: float = 0.0) -> datetime:
    base = datetime(2024, 1, 1, 0, 0, 0)
    return base + timedelta(seconds=offset_seconds)


def test_iceberg_detection_from_quote_refills() -> None:
    detector = HiddenFlowDetector(
        flicker_window_seconds=1.0,
        iceberg_min_refills=1,
        iceberg_partial_fill_threshold=0.25,
    )

    detector.record_quote(
        "XYZ",
        _ts(0.0),
        bid_price=100.0,
        bid_size=1000.0,
        ask_price=100.1,
        ask_size=800.0,
    )
    detector.record_quote(
        "XYZ",
        _ts(0.2),
        bid_price=100.0,
        bid_size=200.0,
        ask_price=100.1,
        ask_size=800.0,
    )
    detector.record_quote(
        "XYZ",
        _ts(0.4),
        bid_price=100.0,
        bid_size=1100.0,
        ask_price=100.1,
        ask_size=800.0,
    )

    detector.record_fill("XYZ", _ts(0.5), price=100.0, quantity=50.0)

    signal = detector.evaluate_symbol("XYZ")
    assert signal.iceberg_detected
    assert not signal.block_trade_detected
    assert not signal.dark_pool_detected
    assert signal.flicker_intensity > 0.0


def test_block_trade_detection_spike() -> None:
    detector = HiddenFlowDetector(
        block_trade_multiplier=3.0,
        block_trade_min_quantity=5.0,
        block_trade_score_threshold=0.2,
    )

    for idx, qty in enumerate((10.0, 11.0, 9.5), start=1):
        detector.record_fill("ABC", _ts(idx), price=50.0, quantity=qty)

    detector.record_fill("ABC", _ts(5), price=50.0, quantity=50.0)

    signal = detector.evaluate_symbol("ABC")
    assert signal.block_trade_detected
    assert signal.block_trade_events == 1
    assert not signal.iceberg_detected
    assert not signal.dark_pool_detected


def test_dark_pool_detection_quiet_midpoint_fill() -> None:
    detector = HiddenFlowDetector(dark_pool_quiet_seconds=0.25)

    detector.record_quote(
        "LMN",
        _ts(0.0),
        bid_price=200.0,
        bid_size=500.0,
        ask_price=200.2,
        ask_size=600.0,
    )

    detector.record_fill("LMN", _ts(0.5), price=200.1, quantity=20.0)

    signal = detector.evaluate_symbol("LMN")
    assert signal.dark_pool_detected
    assert signal.dark_pool_events == 1
    assert not signal.block_trade_detected
    assert not signal.iceberg_detected
