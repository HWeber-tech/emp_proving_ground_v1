from datetime import datetime, timedelta, timezone

import pandas as pd

from src.sensory.how.how_sensor import HowSensor
from src.sensory.how.ict_patterns import ICTPatternAnalyzer, ICTPatternSnapshot


def _build_price_frame() -> pd.DataFrame:
    base = datetime(2024, 5, 1, tzinfo=timezone.utc)
    rows: list[dict[str, object]] = []
    template = {
        "symbol": "EURUSD",
        "volume": 1_000,
        "volatility": 0.0008,
        "spread": 0.00005,
        "depth": 5_000,
        "order_imbalance": 0.1,
        "data_quality": 0.95,
    }

    prices = [
        (1.0950, 1.0995, 1.0940, 1.0980),
        (1.0985, 1.1025, 1.0975, 1.1010),
        (1.1015, 1.1022, 1.1005, 1.1020),
        (1.1021, 1.1080, 1.1018, 1.1075),
        (1.1045, 1.1090, 1.1035, 1.1050),
    ]

    for idx, (open_px, high_px, low_px, close_px) in enumerate(prices):
        payload = dict(template)
        payload.update(
            {
                "timestamp": base + timedelta(minutes=idx),
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": close_px,
            }
        )
        rows.append(payload)

    return pd.DataFrame(rows)


def test_ict_analyzer_detects_bullish_gap_and_sweep() -> None:
    frame = _build_price_frame()
    analyzer = ICTPatternAnalyzer()

    snapshot = analyzer.evaluate(frame)
    assert isinstance(snapshot, ICTPatternSnapshot)
    assert snapshot.bullish_fvg is True
    assert snapshot.bullish_gap_size > 0
    assert snapshot.bearish_fvg is False
    assert snapshot.liquidity_sweep_up is True
    assert snapshot.liquidity_sweep_down is False


def test_how_sensor_emits_ict_telemetry() -> None:
    frame = _build_price_frame()
    sensor = HowSensor()

    signals = sensor.process(frame)
    assert len(signals) == 1
    signal = signals[0]

    metadata = signal.metadata or {}
    ict_metadata = metadata.get("ict_patterns")
    assert isinstance(ict_metadata, dict)
    assert ict_metadata.get("bullish_fvg") is True
    assert ict_metadata.get("liquidity_sweep_up") is True

    value = signal.value
    assert "ict_bullish_fvg" in value
    assert "ict_liquidity_sweep_up" in value
