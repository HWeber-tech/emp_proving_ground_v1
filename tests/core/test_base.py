from datetime import datetime, timezone

import pytest

from src.core.base import DimensionalReading, MarketData


def test_dimensional_reading_value_setter_updates_signal_strength() -> None:
    reading = DimensionalReading(dimension="alpha", signal_strength=0.1)

    reading.value = 1.25

    assert pytest.approx(reading.signal_strength) == 1.25
    assert pytest.approx(reading.value) == 1.25


def test_dimensional_reading_value_setter_rejects_non_numeric() -> None:
    reading = DimensionalReading(dimension="beta", signal_strength=0.3)

    with pytest.raises(ValueError):
        reading.value = "invalid"  # type: ignore[assignment]

def test_dimensional_reading_value_setter_rejects_non_finite() -> None:
    reading = DimensionalReading(dimension="gamma", signal_strength=0.7)

    with pytest.raises(ValueError):
        reading.value = float('nan')
    with pytest.raises(ValueError):
        reading.value = float('inf')


def test_market_data_accepts_iso8601_timestamp() -> None:
    md = MarketData(symbol="EURUSD", timestamp="2023-07-02T15:30:00Z")

    assert md.timestamp == datetime(2023, 7, 2, 15, 30, tzinfo=timezone.utc)


def test_market_data_accepts_lowercase_utc_designator() -> None:
    md = MarketData(symbol="EURUSD", timestamp="2023-07-02T15:30:00z")

    assert md.timestamp == datetime(2023, 7, 2, 15, 30, tzinfo=timezone.utc)
