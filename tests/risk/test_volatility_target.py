from __future__ import annotations

import numpy as np
import pytest

from src.risk.analytics import (
    calculate_realised_volatility,
    determine_target_allocation,
)
from src.risk.analytics.volatility_target import VolatilityTargetAllocation
from src.risk.risk_manager_impl import RiskManagerImpl


def test_calculate_realised_volatility_respects_window() -> None:
    returns = [0.01, -0.02, 0.015, -0.005, 0.007]
    realised = calculate_realised_volatility(returns, window=3, annualisation_factor=1.0)
    expected = float(np.std(np.array(returns[-3:]), ddof=1))
    assert realised == pytest.approx(expected)


def test_determine_target_allocation_caps_leverage() -> None:
    allocation = determine_target_allocation(
        capital=1_000_000,
        target_volatility=0.15,
        realised_volatility=0.02,
        max_leverage=3.0,
    )

    assert isinstance(allocation, VolatilityTargetAllocation)
    assert allocation.leverage == pytest.approx(3.0)
    assert allocation.target_notional == pytest.approx(3_000_000)


def test_risk_manager_volatility_allocation_uses_overrides() -> None:
    manager = RiskManagerImpl(initial_balance=250_000.0)
    returns = [0.012, -0.006, 0.01, -0.008, 0.009, -0.005]

    allocation = manager.target_allocation_from_volatility(
        returns,
        target_volatility=0.2,
        max_leverage=2.5,
        annualisation_factor=1.0,
    )

    realised = calculate_realised_volatility(
        returns, annualisation_factor=1.0
    )
    assert allocation.realised_volatility == pytest.approx(realised)
    assert allocation.target_notional > 0.0
    assert allocation.target_notional <= 250_000 * 2.5 + 1e-6
    assert allocation.volatility_regime is not None
    assert allocation.risk_multiplier is not None
