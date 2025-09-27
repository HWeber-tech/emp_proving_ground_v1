from __future__ import annotations

import pytest

from src.risk.sizing import (
    check_classification_limits,
    compute_classified_exposure,
    kelly_position_size,
    volatility_target_position_size,
)


def test_kelly_position_size_respects_drawdown_caps() -> None:
    size = kelly_position_size(
        equity=7_500.0,
        risk_fraction=0.02,
        stop_loss=0.01,
        win_rate=0.65,
        payoff_ratio=2.0,
        drawdown_multiplier=0.25,
        min_size=1_000.0,
        max_size=20_000.0,
    )

    assert size == pytest.approx(1_781.25, rel=1e-6)


@pytest.mark.parametrize(
    "scenario, capital, target_vol, realized, garch, max_leverage, expected",
    [
        ("tier0", 100_000.0, 0.10, 0.25, None, 5.0, 40_000.0),
        ("tier1", 250_000.0, 0.12, 0.08, 0.10, 4.0, 333_333.3333333333),
    ],
)
def test_volatility_target_position_size_scenarios(
    scenario: str,
    capital: float,
    target_vol: float,
    realized: float | None,
    garch: float | None,
    max_leverage: float,
    expected: float,
) -> None:
    size = volatility_target_position_size(
        capital,
        target_vol,
        realized_volatility=realized,
        garch_volatility=garch,
        max_leverage=max_leverage,
    )

    assert size == pytest.approx(expected, rel=1e-6)


def test_classification_limits_flag_breaches() -> None:
    exposures = {"EURUSD": 2.5, "GBPUSD": 1.0, "AAPL": 0.9}
    classifications = {
        "EURUSD": {"sector": "fx", "asset_class": "currencies"},
        "GBPUSD": {"sector": "fx", "asset_class": "currencies"},
        "AAPL": {"sector": "technology", "asset_class": "equities"},
    }

    sector_totals = compute_classified_exposure(
        exposures, classifications, classification_key="sector"
    )
    assert sector_totals["fx"] == pytest.approx(3.5)

    breaches = check_classification_limits(
        exposures,
        classifications,
        {"fx": 3.0, "technology": 1.5},
        classification_key="sector",
    )

    assert "fx" in breaches
    assert "technology" not in breaches
