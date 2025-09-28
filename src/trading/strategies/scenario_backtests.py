"""Scenario backtesting helpers for roadmap strategy validation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from src.trading.strategies.models import StrategySignal

from .catalog_loader import (
    StrategyCatalog,
    StrategyDefinition,
    instantiate_strategy,
)

__all__ = [
    "MarketScenario",
    "StrategyBacktestResult",
    "DEFAULT_SCENARIOS",
    "run_catalog_backtests",
]


@dataclass(slots=True)
class MarketScenario:
    """Simplified market data snapshot used for deterministic backtests."""

    scenario_id: str
    description: str
    symbol: str
    closes: Sequence[float]
    timeframes: Mapping[str, Sequence[float]] | None = None

    def price_return(self) -> float:
        closes = np.asarray(self.closes, dtype=float)
        if closes.size < 2:
            return 0.0
        start = closes[0]
        end = closes[-1]
        if start == 0.0:
            return 0.0
        return float((end / start) - 1.0)

    def to_market_payload(self) -> dict[str, Mapping[str, object]]:
        payload: dict[str, object] = {"close": list(self.closes)}
        if self.timeframes:
            payload["timeframes"] = {
                timeframe: {"close": list(values)}
                for timeframe, values in self.timeframes.items()
            }
        return {self.symbol: payload}


@dataclass(slots=True)
class StrategyBacktestResult:
    """Summary of a single strategy evaluated on a scenario."""

    strategy_id: str
    scenario_id: str
    expected_return: float
    baseline_return: float
    uplift: float
    direction: str
    leverage: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "strategy_id": self.strategy_id,
            "scenario_id": self.scenario_id,
            "expected_return": self.expected_return,
            "baseline_return": self.baseline_return,
            "uplift": self.uplift,
            "direction": self.direction,
            "leverage": self.leverage,
        }


@dataclass(slots=True)
class _BaselineMAStrategy:
    identifier: str
    capital: float
    short_window: int = 5
    long_window: int = 20
    risk_fraction: float = 0.2

    async def generate_signal(
        self, market_data: Mapping[str, object], symbol: str
    ) -> StrategySignal:
        payload = market_data.get(symbol)
        if not isinstance(payload, Mapping) or "close" not in payload:
            return StrategySignal(
                symbol=symbol,
                action="FLAT",
                confidence=0.0,
                notional=0.0,
                metadata={},
            )

        closes = np.asarray(payload["close"], dtype=float)
        closes = closes[np.isfinite(closes)]
        if closes.size < max(self.short_window, self.long_window):
            return StrategySignal(
                symbol=symbol,
                action="FLAT",
                confidence=0.0,
                notional=0.0,
                metadata={"reason": "insufficient_data"},
            )

        short_ma = float(np.mean(closes[-self.short_window :]))
        long_ma = float(np.mean(closes[-self.long_window :]))
        if short_ma > long_ma:
            action = "BUY"
        elif short_ma < long_ma:
            action = "SELL"
        else:
            action = "FLAT"

        notional = self.capital * float(self.risk_fraction)
        if action == "SELL":
            notional *= -1.0
        confidence = 0.5 if action != "FLAT" else 0.0

        return StrategySignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            notional=notional if action != "FLAT" else 0.0,
            metadata={
                "short_ma": short_ma,
                "long_ma": long_ma,
                "risk_fraction": self.risk_fraction,
            },
        )


def _signal_return(
    signal: StrategySignal, scenario: MarketScenario, capital: float
) -> tuple[float, float]:
    closes = np.asarray(scenario.closes, dtype=float)
    if closes.size >= 2 and closes[-2] != 0.0:
        price_ret = float((closes[-1] / closes[-2]) - 1.0)
    else:
        price_ret = scenario.price_return()
    if signal.action == "BUY":
        direction = 1.0
    elif signal.action == "SELL":
        direction = -1.0
    else:
        return 0.0, 0.0

    if capital <= 0.0:
        return 0.0, direction

    leverage = abs(signal.notional) / capital if capital else 0.0
    expected = direction * price_ret * leverage
    return expected, direction


def _build_entry_payload(scenario: MarketScenario) -> dict[str, Mapping[str, object]]:
    closes = list(scenario.closes)
    if len(closes) <= 1:
        return scenario.to_market_payload()

    entry_closes = closes[:-1]
    if not entry_closes:
        return scenario.to_market_payload()

    payload: dict[str, object] = {"close": entry_closes}
    if scenario.timeframes:
        timeframe_payload: dict[str, object] = {}
        for timeframe, values in scenario.timeframes.items():
            values_list = list(values)
            if len(values_list) > 1:
                truncated = values_list[:-1]
                if truncated:
                    timeframe_payload[timeframe] = {"close": truncated}
            elif values_list:
                timeframe_payload[timeframe] = {"close": values_list}
        if timeframe_payload:
            payload["timeframes"] = timeframe_payload
    return {scenario.symbol: payload}


async def _evaluate_strategy(
    definition: StrategyDefinition, scenario: MarketScenario
) -> StrategyBacktestResult:
    strategy = instantiate_strategy(definition)
    entry_payload = _build_entry_payload(scenario)
    signal = await strategy.generate_signal(entry_payload, scenario.symbol)
    expected_return, direction = _signal_return(
        signal, scenario, definition.capital
    )

    baseline_strategy = _BaselineMAStrategy(
        identifier="baseline",
        capital=definition.capital,
    )
    baseline_signal = await baseline_strategy.generate_signal(
        entry_payload, scenario.symbol
    )
    baseline_return, _ = _signal_return(
        baseline_signal, scenario, baseline_strategy.capital
    )
    uplift = expected_return - baseline_return

    direction_label = "BUY" if direction >= 0 else "SELL"
    leverage = abs(signal.notional) / definition.capital if definition.capital else 0.0

    return StrategyBacktestResult(
        strategy_id=definition.identifier,
        scenario_id=scenario.scenario_id,
        expected_return=expected_return,
        baseline_return=baseline_return,
        uplift=uplift,
        direction=direction_label,
        leverage=leverage,
    )


async def run_catalog_backtests(
    catalog: StrategyCatalog, scenarios: Iterable[MarketScenario]
) -> list[StrategyBacktestResult]:
    results: list[StrategyBacktestResult] = []
    for definition in catalog.enabled_strategies():
        for scenario in scenarios:
            result = await _evaluate_strategy(definition, scenario)
            results.append(result)
    return results


DEFAULT_SCENARIOS: tuple[MarketScenario, ...] = (
    MarketScenario(
        scenario_id="trend_bull",
        description="Persistent uptrend with low realised volatility",
        symbol="EURUSD",
        closes=[
            1.000,
            1.003,
            1.006,
            1.010,
            1.015,
            1.021,
            1.028,
            1.036,
            1.045,
            1.055,
        ],
        timeframes={
            "1h": [
                1.000,
                1.001,
                1.002,
                1.004,
                1.006,
                1.009,
                1.012,
                1.016,
                1.021,
                1.027,
            ],
            "15m": [
                1.0000,
                1.0003,
                1.0008,
                1.0014,
                1.0021,
                1.0030,
                1.0040,
                1.0052,
                1.0065,
                1.0079,
            ],
        },
    ),
    MarketScenario(
        scenario_id="trend_bear",
        description="Sustained downtrend testing sell-side conviction",
        symbol="EURUSD",
        closes=[
            1.120,
            1.115,
            1.109,
            1.102,
            1.094,
            1.085,
            1.075,
            1.064,
            1.052,
            1.039,
        ],
        timeframes={
            "1h": [
                1.120,
                1.118,
                1.115,
                1.111,
                1.106,
                1.100,
                1.093,
                1.085,
                1.076,
                1.066,
            ],
            "15m": [
                1.1200,
                1.1195,
                1.1187,
                1.1176,
                1.1162,
                1.1145,
                1.1125,
                1.1102,
                1.1075,
                1.1045,
            ],
        },
    ),
    MarketScenario(
        scenario_id="mean_reversion",
        description="Price overshoot followed by reversion to the mean",
        symbol="EURUSD",
        closes=[
            1.050,
            1.050,
            1.050,
            1.050,
            1.050,
            1.050,
            1.080,
            1.095,
            1.110,
            1.135,
            1.150,
            1.085,
        ],
    ),
    MarketScenario(
        scenario_id="volatility_breakout",
        description="Compression then explosive breakout",
        symbol="EURUSD",
        closes=[
            1.000,
            1.000,
            1.000,
            1.000,
            1.000,
            1.000,
            1.000,
            1.000,
            1.000,
            1.000,
            1.000,
            1.000,
            1.000,
            1.000,
            1.005,
            1.010,
            1.030,
            1.080,
            1.140,
        ],
    ),
)


if __name__ == "__main__":
    from .catalog_loader import load_strategy_catalog

    async def _run() -> None:
        catalog = load_strategy_catalog()
        results = await run_catalog_backtests(catalog, DEFAULT_SCENARIOS)
        for result in results:
            print(result.as_dict())

    asyncio.run(_run())
