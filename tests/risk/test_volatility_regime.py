from __future__ import annotations

from src.config.risk.risk_config import RiskConfig
from src.risk.analytics import (
    VolatilityRegime,
    classify_volatility_regime,
)
from src.risk.risk_manager_impl import RiskManagerImpl


def _generate_returns(scale: float, count: int = 50) -> list[float]:
    # Deterministic alternating sequence to avoid random flakiness
    return [(((-1) ** i) * scale) for i in range(count)]


def test_classify_volatility_regime_extreme() -> None:
    returns = _generate_returns(0.08)
    assessment = classify_volatility_regime(
        returns,
        target_volatility=0.15,
        annualisation_factor=1.0,
    )
    assert assessment.regime is VolatilityRegime.EXTREME
    assert assessment.risk_multiplier < 1.0
    assert assessment.realised_volatility >= assessment.thresholds.extreme


def test_classify_volatility_regime_low() -> None:
    returns = _generate_returns(0.002)
    assessment = classify_volatility_regime(
        returns,
        target_volatility=0.15,
        annualisation_factor=1.0,
    )
    assert assessment.regime is VolatilityRegime.LOW
    assert assessment.risk_multiplier > 1.0
    assert assessment.realised_volatility <= assessment.thresholds.low


def test_risk_manager_allocation_includes_regime_signal() -> None:
    manager = RiskManagerImpl(initial_balance=100_000.0, risk_config=RiskConfig())
    high_vol_returns = _generate_returns(0.06)
    allocation = manager.target_allocation_from_volatility(
        high_vol_returns,
        target_volatility=0.2,
        annualisation_factor=1.0,
        max_leverage=2.0,
    )
    assert allocation.volatility_regime in {
        VolatilityRegime.HIGH.value,
        VolatilityRegime.EXTREME.value,
    }
    assert allocation.risk_multiplier is not None
    assert allocation.leverage <= 2.0

    low_vol_returns = _generate_returns(0.0015)
    allocation_low = manager.target_allocation_from_volatility(
        low_vol_returns,
        target_volatility=0.2,
        annualisation_factor=1.0,
        max_leverage=2.0,
    )
    assert allocation_low.volatility_regime == VolatilityRegime.LOW.value
    assert allocation_low.risk_multiplier is not None
    assert allocation_low.leverage <= 2.0
    assert allocation_low.leverage >= allocation.leverage
