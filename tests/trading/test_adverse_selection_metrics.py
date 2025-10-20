from __future__ import annotations

import pytest

from src.trading.risk import MicropriceDriftResult, compute_microprice_drift


def test_microprice_drift_computes_directional_stats() -> None:
    events = [
        {"microprice": 100.00, "our_action": False},
        {"microprice": 100.10, "our_action": True, "side": "BUY"},
        {"microprice": 100.12, "our_action": False},
        {"microprice": 100.08, "our_action": True, "side": "sell"},
        {"microprice": 100.06, "our_action": False},
        {"microprice": 100.03, "our_action": False},
    ]

    result = compute_microprice_drift(events, horizon_events=2)

    assert isinstance(result, MicropriceDriftResult)
    assert result.samples == 2
    assert result.mean_drift == pytest.approx(-0.035)
    assert result.median_drift == pytest.approx(-0.035)
    assert result.positive_fraction == pytest.approx(0.0)
    assert result.mean_by_side["buy"] == pytest.approx(-0.02)
    assert result.mean_by_side["sell"] == pytest.approx(-0.05)
    assert result.mean_by_side["unknown"] is None
    assert result.count_by_side == {"buy": 1, "sell": 1, "unknown": 0}


def test_microprice_drift_infers_action_from_action_field() -> None:
    events = [
        {"microprice": 101.0, "action": "BUY"},
        {"microprice": 101.05},
        {"microprice": 101.12, "action": "sell"},
        {"microprice": 101.20},
    ]

    result = compute_microprice_drift(events, horizon_events=1)

    assert result.samples == 2
    assert result.mean_drift == pytest.approx(0.065)
    assert result.mean_by_side["buy"] == pytest.approx(0.05)
    assert result.mean_by_side["sell"] == pytest.approx(0.08)
    assert result.positive_fraction == pytest.approx(1.0)


def test_microprice_drift_requires_positive_horizon() -> None:
    with pytest.raises(ValueError):
        compute_microprice_drift([], horizon_events=0)
