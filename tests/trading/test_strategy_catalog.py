import asyncio
from pathlib import Path

import pytest

from src.trading.strategies.catalog_loader import (
    StrategyCatalog,
    instantiate_strategy,
    load_strategy_catalog,
)


def test_load_strategy_catalog_defaults(tmp_path: Path) -> None:
    catalog_path = Path("config/trading/strategy_catalog.yaml")
    catalog = load_strategy_catalog(catalog_path)

    assert isinstance(catalog, StrategyCatalog)
    assert catalog.version
    assert catalog.default_capital > 0
    assert catalog.enabled_strategies()

    momentum = next(
        (definition for definition in catalog.definitions if definition.key == "momentum"),
        None,
    )
    assert momentum is not None
    assert momentum.class_name == "MomentumStrategy"
    assert momentum.symbols


def test_instantiate_strategy_returns_correct_class() -> None:
    catalog = load_strategy_catalog()
    definitions = catalog.enabled_strategies()
    assert definitions

    strategy_objects = [instantiate_strategy(defn) for defn in definitions]
    identifiers = {strategy.strategy_id for strategy in strategy_objects}

    expected_ids = {definition.identifier for definition in definitions}
    assert identifiers == expected_ids

    for definition, strategy in zip(definitions, strategy_objects):
        assert strategy.symbols
        assert strategy.strategy_id == definition.identifier


@pytest.mark.asyncio
async def test_catalog_backtests_produce_results() -> None:
    from src.trading.strategies.scenario_backtests import (
        DEFAULT_SCENARIOS,
        run_catalog_backtests,
    )

    catalog = load_strategy_catalog()
    results = await run_catalog_backtests(catalog, DEFAULT_SCENARIOS)

    assert results
    scenario_ids = {result.scenario_id for result in results}
    assert scenario_ids == {scenario.scenario_id for scenario in DEFAULT_SCENARIOS}

    for result in results:
        assert isinstance(result.strategy_id, str)
        assert isinstance(result.expected_return, float)
        assert result.direction in {"BUY", "SELL"}
        assert result.leverage >= 0.0
