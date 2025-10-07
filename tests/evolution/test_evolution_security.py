"""Security hardening regression tests for the evolution engine helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping

import pytest

from src.core.evolution.engine import EvolutionConfig, EvolutionEngine
from src.core.evolution.seeding import GenomeSeed, apply_seed_to_genome


@dataclass
class UncooperativeGenome:
    """Genome stub whose ``with_updated`` helper raises to exercise logging."""

    id: str = "genome-updater"
    parent_ids: list[str] = field(default_factory=list)
    generation: int = 0

    def with_updated(self, **_: object) -> "UncooperativeGenome":
        raise TypeError("cannot update frozen genome")


class FailingNormalizerGenome:
    """Genome stub whose normaliser raises to verify guardrail logging."""

    id = "genome-normalizer"

    def __init__(self) -> None:
        self._normalize_weights = self._normalize  # type: ignore[attr-defined]

    def _normalize(self) -> None:
        raise ValueError("normalizer failed")


@dataclass(frozen=True)
class FrozenGenome:
    """Frozen genome stub that rejects attribute mutation."""

    id: str = "genome-frozen"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def with_updated(self, **_: object) -> "FrozenGenome":
        raise TypeError("immutable genome")


def _has_message(records: list[logging.LogRecord], fragment: str) -> bool:
    return any(fragment in record.getMessage() for record in records)


def test_apply_parent_metadata_logs_when_with_updated_fails(caplog: pytest.LogCaptureFixture) -> None:
    engine = EvolutionEngine(EvolutionConfig(population_size=1))
    genome = UncooperativeGenome()

    with caplog.at_level(logging.WARNING):
        result = engine._apply_parent_metadata(genome, ["parent-1"], 7)

    assert result is genome
    assert genome.parent_ids == ["parent-1"]
    assert genome.generation == 7
    assert _has_message(caplog.records, "Applying parent metadata via with_updated")


def test_normalize_genome_logs_guardrails(caplog: pytest.LogCaptureFixture) -> None:
    engine = EvolutionEngine(EvolutionConfig(population_size=1))
    genome = FailingNormalizerGenome()

    with caplog.at_level(logging.WARNING):
        engine._normalize_genome(genome)

    assert _has_message(caplog.records, "Normalizing genome weights")


def test_apply_seed_to_genome_logs_on_attribute_failures(caplog: pytest.LogCaptureFixture) -> None:
    seed = GenomeSeed(
        name="seed",
        species="spec",
        parameters={},
        parent_ids=("p1",),
        mutation_history=("m1",),
        performance_metrics={"return": 1.0},
        tags=("tag",),
    )
    genome = FrozenGenome()

    with caplog.at_level(logging.WARNING):
        result = apply_seed_to_genome(genome, seed)

    assert result is genome
    assert _has_message(caplog.records, "Seed hardening: applying seed updates via with_updated")
    assert _has_message(caplog.records, "Seed hardening: setting attribute 'parent_ids'")
    assert _has_message(caplog.records, "Seed hardening: merging seed metadata")
