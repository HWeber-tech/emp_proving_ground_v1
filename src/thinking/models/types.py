from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Protocol, TypedDict, runtime_checkable


@runtime_checkable
class PredictionLike(Protocol):
    confidence: float
    probability: float


@runtime_checkable
class SurvivalResultLike(Protocol):
    survival_rate: float


@runtime_checkable
class RedTeamResultLike(Protocol):
    survival_probability: float
    weaknesses: Sequence[object]


class AttackReportTD(TypedDict, total=False):
    attack_id: str
    strategy_id: str
    success: bool
    impact: float
    timestamp: str
    error: str


@runtime_checkable
class AlgorithmSignatureLike(Protocol):
    algorithm_type: str
    confidence: Decimal | float
    frequency: str


@runtime_checkable
class CompetitorBehaviorLike(Protocol):
    competitor_id: str
    algorithm_signature: AlgorithmSignatureLike
    behavior_metrics: dict[str, float]
    patterns: Sequence[str]
    threat_level: str
    market_share: Decimal
    performance: Decimal


@runtime_checkable
class CounterStrategyLike(Protocol):
    strategy_id: str
    target_competitor: str
    counter_type: str
    parameters: dict[str, object]
    expected_effectiveness: Decimal


class UnderstandingReportTD(TypedDict, total=False):
    """Telemetry snapshot emitted by the competitive understanding system.

    The structure retains the legacy ``intelligence_id`` field so downstream
    callers that have not yet migrated continue to function while the new
    ``understanding_id`` becomes the canonical identifier.
    """

    understanding_id: str
    intelligence_id: str  # Backwards-compat alias (legacy field)
    signatures_detected: int
    competitors_analyzed: int
    counter_strategies_developed: int
    market_share_analysis: dict[str, object]
    timestamp: str


# Backwards-compatible alias while the namespace transition continues.
IntelligenceReportTD = UnderstandingReportTD


__all__ = [
    "PredictionLike",
    "SurvivalResultLike",
    "RedTeamResultLike",
    "AttackReportTD",
    "UnderstandingReportTD",
    "IntelligenceReportTD",
    "AlgorithmSignatureLike",
    "CompetitorBehaviorLike",
    "CounterStrategyLike",
]
