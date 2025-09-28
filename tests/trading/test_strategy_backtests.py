import pytest

from src.trading.strategies.catalog_loader import load_strategy_catalog
from src.trading.strategies.scenario_backtests import (
    DEFAULT_SCENARIOS,
    run_catalog_backtests,
)


@pytest.mark.asyncio
async def test_momentum_outperforms_baseline_in_trend() -> None:
    catalog = load_strategy_catalog()
    scenarios = [scenario for scenario in DEFAULT_SCENARIOS if scenario.scenario_id == "trend_bull"]
    assert scenarios

    results = await run_catalog_backtests(catalog, scenarios)
    momentum_result = next(result for result in results if result.strategy_id == "momentum_v1")

    assert momentum_result.expected_return > momentum_result.baseline_return
    assert momentum_result.uplift > 0


@pytest.mark.asyncio
async def test_mean_reversion_handles_reversion_scenario() -> None:
    catalog = load_strategy_catalog()
    scenarios = [scenario for scenario in DEFAULT_SCENARIOS if scenario.scenario_id == "mean_reversion"]
    results = await run_catalog_backtests(catalog, scenarios)

    mean_rev_result = next(result for result in results if result.strategy_id == "mean_rev_v1")
    assert mean_rev_result.expected_return > mean_rev_result.baseline_return


@pytest.mark.asyncio
async def test_breakout_strategy_accelerates_in_volatility_spike() -> None:
    catalog = load_strategy_catalog()
    scenarios = [scenario for scenario in DEFAULT_SCENARIOS if scenario.scenario_id == "volatility_breakout"]
    results = await run_catalog_backtests(catalog, scenarios)

    breakout_result = next(result for result in results if result.strategy_id == "vol_break_v1")
    assert breakout_result.expected_return > breakout_result.baseline_return
