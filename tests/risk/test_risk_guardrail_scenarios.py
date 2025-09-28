"""Tier-0/Tier-1 risk guardrail scenarios derived from the roadmap."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Iterable

import pytest

from src.config.risk.risk_config import RiskConfig
from src.risk.risk_manager_impl import RiskManagerImpl


@pytest.fixture
def baseline_risk_config() -> RiskConfig:
    """Return a RiskConfig tuned for guardrail scenario evaluation."""

    return RiskConfig(
        max_risk_per_trade_pct=Decimal("0.05"),
        max_total_exposure_pct=Decimal("0.30"),
        instrument_sector_map={
            "TECH_ALPHA": "TECH",
            "TECH_BETA": "TECH",
            "FX_EURUSD": "FX",
        },
        sector_exposure_limits={
            "TECH": Decimal("0.05"),
            "FX": Decimal("0.10"),
        },
        min_position_size=1000,
    )


TierScenario = Dict[str, Any]


def _instantiate_manager(
    config: RiskConfig, preload: Iterable[Dict[str, Any]]
) -> RiskManagerImpl:
    manager = RiskManagerImpl(initial_balance=100_000.0, risk_config=config)
    for entry in preload:
        manager.add_position(
            entry["symbol"],
            entry["size"],
            entry["entry_price"],
            stop_loss_pct=entry.get("stop_loss_pct"),
        )
    return manager


TIER0_SCENARIOS: tuple[TierScenario, ...] = (
    {
        "id": "fx_micro_position",
        "position": {
            "symbol": "FX_EURUSD",
            "size": 1500,
            "entry_price": 1.10,
            "stop_loss_pct": 0.01,
        },
        "preload": (),
    },
    {
        "id": "tech_equity_within_sector_budget",
        "position": {
            "symbol": "TECH_ALPHA",
            "size": 1800,
            "entry_price": 40.0,
            "stop_loss_pct": 0.02,
        },
        "preload": (
            {
                "symbol": "TECH_BETA",
                "size": 1500,
                "entry_price": 38.0,
                "stop_loss_pct": 0.02,
            },
        ),
    },
)


TIER1_SCENARIOS: tuple[TierScenario, ...] = (
    {
        "id": "missing_stop_loss",
        "position": {
            "symbol": "TECH_ALPHA",
            "size": 1500,
            "entry_price": 35.0,
        },
        "preload": (),
    },
    {
        "id": "sector_exposure_breach",
        "position": {
            "symbol": "TECH_ALPHA",
            "size": 2000,
            "entry_price": 50.0,
            "stop_loss_pct": 0.05,
        },
        "preload": (
            {
                "symbol": "TECH_BETA",
                "size": 2200,
                "entry_price": 45.0,
                "stop_loss_pct": 0.05,
            },
        ),
    },
)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", TIER0_SCENARIOS, ids=lambda s: s["id"])
async def test_tier0_scenarios_pass(
    scenario: TierScenario, baseline_risk_config: RiskConfig
) -> None:
    """Tier-0 encyclopedia scenarios must clear guardrails."""

    manager = _instantiate_manager(
        baseline_risk_config.copy(deep=True), scenario["preload"]
    )
    assert await manager.validate_position(dict(scenario["position"])) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", TIER1_SCENARIOS, ids=lambda s: s["id"])
async def test_tier1_scenarios_fail(
    scenario: TierScenario, baseline_risk_config: RiskConfig
) -> None:
    """Tier-1 encyclopedia scenarios should trigger guardrails."""

    manager = _instantiate_manager(
        baseline_risk_config.copy(deep=True), scenario["preload"]
    )
    assert await manager.validate_position(dict(scenario["position"])) is False
