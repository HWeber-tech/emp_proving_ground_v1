"""Regression tests for hardened ecosystem optimizer behaviours."""

from __future__ import annotations

import logging
import random
from typing import Iterator

import pytest

from src.ecosystem.optimization.ecosystem_optimizer import EcosystemOptimizer
from src.genome.models.genome import DecisionGenome as CanonDecisionGenome


class _FailingRegime:
    def __str__(self) -> str:  # pragma: no cover - invoked indirectly
        raise ValueError("regime cannot be stringified")


class _MarketContextStub:
    def __init__(self, regime: object) -> None:
        self.regime = regime
        self.data: dict[str, object] = {}


@pytest.mark.asyncio()
async def test_evaluate_genome_performance_handles_non_stringifiable_regime() -> None:
    optimizer = EcosystemOptimizer()
    genome = CanonDecisionGenome.from_dict(
        {
            "id": "g-1",
            "parameters": {"alpha": 1.0},
            "species_type": "stalker",
        }
    )

    score = await optimizer._evaluate_genome_performance(  # type: ignore[attr-defined]
        genome,
        _MarketContextStub(_FailingRegime()),
    )

    assert isinstance(score, float)


def test_crossover_skips_non_numeric_parameters() -> None:
    optimizer = EcosystemOptimizer()
    parent1 = CanonDecisionGenome(
        id="p1",
        parameters={"alpha": 1.0, "beta": "invalid"},  # type: ignore[assignment]
        species_type="stalker",
        parent_ids=[],
        mutation_history=[],
        performance_metrics={},
    )
    parent2 = CanonDecisionGenome(
        id="p2",
        parameters={"alpha": 2.0, "beta": object()},  # type: ignore[assignment]
        species_type="stalker",
        parent_ids=[],
        mutation_history=[],
        performance_metrics={},
    )

    child = optimizer._crossover_genomes(parent1, parent2)

    assert "alpha" in child.parameters
    assert "beta" not in child.parameters


def _random_sequence(values: list[float]) -> Iterator[float]:
    for value in values:
        yield value
    while True:
        yield values[-1]


def test_mutate_genome_ignores_non_numeric_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = EcosystemOptimizer()
    genome = CanonDecisionGenome(
        id="g-immutable",
        parameters={"good": 1.0, "bad": "oops"},  # type: ignore[assignment]
        species_type="stalker",
        parent_ids=[],
        mutation_history=[],
        performance_metrics={},
    )

    random_values = _random_sequence([1.0, 0.0])
    monkeypatch.setattr(random, "random", lambda: next(random_values))
    monkeypatch.setattr(random, "uniform", lambda _a, _b: 1.1)

    mutated = optimizer._mutate_genome(genome)

    assert mutated is genome


def test_ensure_canonical_logs_when_adaptation_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    optimizer = EcosystemOptimizer()

    def _raise_type_error(_genome: object) -> CanonDecisionGenome:
        raise TypeError("broken adapter")

    monkeypatch.setattr(
        "src.ecosystem.optimization.ecosystem_optimizer.adapt_to_canonical",
        _raise_type_error,
    )

    with caplog.at_level(logging.WARNING):
        fallback = optimizer._ensure_canonical(object())

    assert fallback.parameters == {}
    assert "Failed to adapt genome" in caplog.text

