#!/usr/bin/env python3
"""
Canonical StrategyTester for strategy evaluation under the trading engine.

This class provides a simple, dependency-light API to evaluate strategies
against market scenarios and report survival and performance metrics.

It is designed to serve as the single source of truth for 'StrategyTester'
and to be consumed by shims in intelligence/thinking layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class CanonicalTestResult:
    """Generic result object with attributes expected by various callers."""

    strategy_id: str
    survived: bool
    performance_score: float
    stress_endurance: float
    adaptation_score: float
    # Optional fields for callers that expect scenario-level attribution
    scenario_id: Optional[str] = None
    timestamp: datetime = datetime.utcnow()


class StrategyTester:
    """
    Canonical strategy tester.

    Public API:
      - test_strategies(strategy_population, scenarios) -> list[CanonicalTestResult]
    """

    def __init__(self, performance_threshold: float = 0.7) -> None:
        self.performance_threshold = performance_threshold

    async def test_strategies(
        self,
        strategy_population: list[dict[str, object]],
        scenarios: list[object],
    ) -> list[CanonicalTestResult]:
        """
        Evaluate each strategy against all scenarios and return canonical results.

        Notes:
          - Scenarios are treated generically. Where possible, scenario_id is
            extracted from attribute 'scenario_id' or 'id' if present.
        """
        results: list[CanonicalTestResult] = []

        total = max(1, len(scenarios))
        for strategy in strategy_population:
            strategy_id = strategy.get("id", "unknown")

            survival_count = 0
            total_performance = 0.0
            last_scenario_id: Optional[str] = None

            for scenario in scenarios:
                last_scenario_id = getattr(scenario, "scenario_id", getattr(scenario, "id", None))
                perf = await self._evaluate_strategy_performance(strategy, scenario)
                if perf > 0:
                    survival_count += 1
                total_performance += perf

            stress_endurance = survival_count / total
            avg_performance = total_performance / total

            result = CanonicalTestResult(
                strategy_id=str(strategy.get("id", "unknown")),
                survived=stress_endurance >= self.performance_threshold,
                performance_score=avg_performance,
                stress_endurance=stress_endurance,
                adaptation_score=self._calculate_adaptation_score(strategy, scenarios),
                scenario_id=last_scenario_id,
                timestamp=datetime.utcnow(),
            )
            results.append(result)

        return results

    async def _evaluate_strategy_performance(
        self, strategy: dict[str, object], scenario: object
    ) -> float:
        """
        Simplified performance estimation. Callers can refine this via
        upstream preparation of 'strategy' and 'scenario' attributes.
        """
        # Prefer explicit performance hints if provided by strategy/scenario
        hint = strategy.get("performance_hint")
        if isinstance(hint, (int, float)):
            return float(max(-1.0, min(1.0, hint)))

        # Generic heuristic: if scenario exposes 'difficulty_level', reduce score
        difficulty = getattr(scenario, "difficulty_level", None)
        if isinstance(difficulty, (int, float)):
            base = 0.02
            penalty = float(difficulty) * 0.1
            return max(-1.0, min(1.0, base - penalty))

        # Fallback constant small positive performance
        return 0.01

    def _calculate_adaptation_score(
        self, strategy: dict[str, object], scenarios: list[object]
    ) -> float:
        """
        Lightweight adaptation score based on strategy features, if any.
        """
        features_obj = strategy.get("adaptation_features")
        features: dict[str, object] = features_obj if isinstance(features_obj, dict) else {}
        score = 0.0
        if features.get("dynamic_risk"):
            score += 0.3
        if features.get("pattern_learning"):
            score += 0.3
        if features.get("regime_detection"):
            score += 0.2
        if features.get("parameter_optimization"):
            score += 0.2
        return float(max(0.0, min(1.0, score)))
