from __future__ import annotations

import pandas as pd
import pytest

from src.sensory.organs.dimensions.order_flow import (
    MarketMicrostructureAnalyzer,
    OrderFlowAnalyzer,
)


def _balanced_fixture() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=6, freq="h")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100, 101, 102, 103, 100, 100],
            "high": [101, 103, 105, 107, 105, 99],
            "low": [99, 100, 102, 100, 98, 95],
            "close": [100, 102, 103, 101, 104, 96],
            "volume": [10_000, 11_000, 12_000, 11_500, 12_500, 12_200],
        }
    )


def _bullish_bias_fixture() -> pd.DataFrame:
    timestamps = pd.date_range("2024-02-01", periods=6, freq="h")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100, 101, 105, 108, 113, 117],
            "high": [101, 105, 109, 113, 118, 122],
            "low": [99, 99, 102, 101, 111, 107],
            "close": [100, 101, 108, 104, 112, 108],
            "volume": [9000, 9400, 9800, 9700, 9600, 9500],
        }
    )


def test_microstructure_detects_fvg_and_liquidity_sweeps() -> None:
    frame = _balanced_fixture()
    analyzer = MarketMicrostructureAnalyzer()

    result = analyzer.analyze_microstructure(frame)

    fvg = result["fair_value_gaps"]
    sweeps = result["liquidity_sweeps"]

    assert fvg["bullish_count"] == 1
    assert fvg["bearish_count"] == 1
    assert sweeps["buy_side"] == 1
    assert sweeps["sell_side"] == 1
    assert fvg["average_gap_size"] == pytest.approx(1.0)
    assert result["microstructure_score"] == pytest.approx(0.0, abs=1e-9)


def test_order_flow_bias_highlights_buy_pressure() -> None:
    frame = _bullish_bias_fixture()
    analyzer = OrderFlowAnalyzer()

    result = analyzer.analyze_institutional_flow(frame)

    pressure = result["institutional_pressure"]
    assert result["flow_strength"] == pytest.approx(1.0)
    assert pressure["buying_pressure"] == pytest.approx(1.0)
    assert pressure["selling_pressure"] == pytest.approx(0.0)

    microstructure = result["microstructure"]
    assert microstructure["fair_value_gaps"]["bullish_count"] == 2
    assert microstructure["liquidity_sweeps"]["buy_side"] == 2
