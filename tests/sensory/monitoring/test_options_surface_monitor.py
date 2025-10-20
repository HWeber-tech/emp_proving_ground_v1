from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.sensory.monitoring import OptionsSurfaceMonitor


def _build_sample_surface() -> pd.DataFrame:
    data = [
        {
            "timestamp": "2024-05-20T14:30:00Z",
            "symbol": "EURUSD",
            "underlying_price": 100.0,
            "strike": 105.0,
            "option_type": "call",
            "implied_volatility": 0.22,
            "open_interest": 200,
            "delta": 0.35,
            "gamma": 0.020,
            "contract_multiplier": 100,
        },
        {
            "timestamp": "2024-05-20T14:30:00Z",
            "symbol": "EURUSD",
            "underlying_price": 100.0,
            "strike": 110.0,
            "option_type": "call",
            "implied_volatility": 0.25,
            "open_interest": 150,
            "delta": 0.25,
            "gamma": 0.030,
            "contract_multiplier": 100,
        },
        {
            "timestamp": "2024-05-20T14:30:00Z",
            "symbol": "EURUSD",
            "underlying_price": 100.0,
            "strike": 100.0,
            "option_type": "call",
            "implied_volatility": 0.21,
            "open_interest": 300,
            "delta": 0.50,
            "gamma": 0.040,
            "contract_multiplier": 100,
        },
        {
            "timestamp": "2024-05-20T14:30:00Z",
            "symbol": "EURUSD",
            "underlying_price": 100.0,
            "strike": 95.0,
            "option_type": "put",
            "implied_volatility": 0.28,
            "open_interest": 180,
            "delta": -0.40,
            "gamma": 0.025,
            "contract_multiplier": 100,
        },
        {
            "timestamp": "2024-05-20T14:30:00Z",
            "symbol": "EURUSD",
            "underlying_price": 100.0,
            "strike": 90.0,
            "option_type": "put",
            "implied_volatility": 0.32,
            "open_interest": 220,
            "delta": -0.50,
            "gamma": 0.020,
            "contract_multiplier": 100,
        },
        {
            "timestamp": "2024-05-20T14:30:00Z",
            "symbol": "EURUSD",
            "underlying_price": 100.0,
            "strike": 100.0,
            "option_type": "put",
            "implied_volatility": 0.26,
            "open_interest": 310,
            "delta": -0.50,
            "gamma": 0.040,
            "contract_multiplier": 100,
        },
    ]
    return pd.DataFrame(data)


def test_options_surface_monitor_summarise_core_metrics():
    surface = _build_sample_surface()
    monitor = OptionsSurfaceMonitor()

    summary = monitor.summarise(surface)

    assert summary.symbol == "EURUSD"
    assert summary.spot_price == pytest.approx(100.0)
    assert summary.metadata["rows"] == len(surface)
    assert summary.as_of.tzinfo == timezone.utc

    assert summary.iv_skew.direction == "put"
    assert summary.iv_skew.skew == pytest.approx(-0.06135, abs=1e-4)

    walls = summary.open_interest_walls
    assert walls
    assert walls[0].strike == pytest.approx(100.0)
    assert walls[0].dominant_side == "put"

    assert summary.delta_imbalance.direction == "put"
    assert summary.delta_imbalance.net_delta == pytest.approx(-7950.0, abs=1e-6)
    assert summary.delta_imbalance.normalised == pytest.approx(-0.1337, abs=1e-4)

    assert summary.gamma_exposure.has_data is True
    assert summary.gamma_exposure.symbol == "EURUSD"


def test_options_surface_monitor_handles_missing_inputs():
    surface = pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "symbol": ["TEST", "TEST"],
            "strike": [100, 105],
            "gamma": [0.01, 0.02],
            "open_interest": [100, 120],
        }
    )
    monitor = OptionsSurfaceMonitor()

    as_of = datetime(2024, 1, 1, tzinfo=timezone.utc)
    summary = monitor.summarise(surface, spot_price=100, symbol="TEST", as_of=as_of)

    assert summary.symbol == "TEST"
    assert summary.iv_skew.skew is None
    assert summary.delta_imbalance.net_delta is None
    strikes = sorted(wall.strike for wall in summary.open_interest_walls)
    assert strikes == [100.0, 105.0]
    assert summary.gamma_exposure.has_data is True
