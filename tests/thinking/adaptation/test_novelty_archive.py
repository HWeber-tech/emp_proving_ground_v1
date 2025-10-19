from __future__ import annotations

from decimal import Decimal

import pytest

from src.thinking.adaptation import (
    NoveltyArchive,
    StrategyExecutionTopology,
    StrategyFeature,
    StrategyGenotype,
    StrategyRiskTemplate,
    compute_genotype_signature,
)
from src.config.risk.risk_config import RiskConfig


def _build_genotype(
    *,
    threshold: float = 0.5,
    floor: float = 0.02,
    latency_budget_ms: int = 50,
    risk_pct: str = "0.02",
    family: str = "momentum",
) -> StrategyGenotype:
    return StrategyGenotype(
        strategy_id="momentum_v1",
        features=(
            StrategyFeature(
                name="trend_strength",
                inputs=("price", "volume"),
                parameters={"window": 20, "threshold": threshold},
            ),
            StrategyFeature(
                name="volatility_floor",
                inputs=("price",),
                parameters={"window": 10, "floor": floor},
            ),
        ),
        execution_topology=StrategyExecutionTopology(
            name="single_leg",
            version="v1",
            parameters={"latency_budget_ms": latency_budget_ms},
        ),
        risk_template=StrategyRiskTemplate(
            template_id="baseline",
            base_config=RiskConfig(),
            overrides={"max_risk_per_trade_pct": Decimal(risk_pct)},
        ),
        metadata={"family": family},
    )


def test_probe_does_not_mutate_archive() -> None:
    archive = NoveltyArchive(capacity=4, dimensions=8)
    genotype = _build_genotype()

    probe = archive.probe(genotype)

    assert pytest.approx(1.0, rel=1e-9) == probe.novelty
    assert len(archive) == 0


def test_record_first_entry_has_max_novelty() -> None:
    archive = NoveltyArchive(capacity=4, dimensions=8)
    genotype = _build_genotype()

    probe = archive.record(genotype)

    assert pytest.approx(1.0, rel=1e-9) == probe.novelty
    assert len(archive) == 1
    assert compute_genotype_signature(genotype) in archive


def test_duplicate_signature_reports_zero_novelty() -> None:
    archive = NoveltyArchive(capacity=4, dimensions=8)
    genotype = _build_genotype()
    archive.record(genotype)

    duplicate = archive.record(_build_genotype())

    assert pytest.approx(0.0, abs=1e-9) == duplicate.novelty
    assert len(archive) == 1


def test_variation_yields_fractional_novelty() -> None:
    archive = NoveltyArchive(capacity=4, dimensions=8)
    baseline = _build_genotype()
    archive.record(baseline)

    varied = _build_genotype(threshold=0.65, floor=0.03, risk_pct="0.03")
    probe = archive.record(varied)

    assert 0.0 < probe.novelty <= 1.0
    assert len(archive) == 2


def test_capacity_eviction_discards_oldest() -> None:
    archive = NoveltyArchive(capacity=2, dimensions=8)
    first = _build_genotype()
    second = _build_genotype(threshold=0.6, floor=0.025, family="momentum-b")
    third = _build_genotype(threshold=0.4, floor=0.018, family="momentum-c")

    archive.record(first)
    archive.record(second)
    archive.record(third)

    assert len(archive) == 2
    assert compute_genotype_signature(first) not in archive
    assert compute_genotype_signature(second) in archive
    assert compute_genotype_signature(third) in archive
