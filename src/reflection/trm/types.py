"""Shared data structures for the TRM production runner."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(slots=True)
class DecisionDiaryEntry:
    """Normalised decision diary payload."""

    raw: Mapping[str, Any]
    timestamp: dt.datetime
    strategy_id: str
    instrument: str
    pnl: float
    risk_flags: tuple[str, ...]
    outcome_labels: tuple[str, ...]
    features_digest: Mapping[str, float]
    belief_confidence: float | None
    action: str
    input_hash: str


@dataclass(slots=True)
class RIMWindow:
    start: dt.datetime
    end: dt.datetime
    minutes: int


@dataclass(slots=True)
class RIMInputBatch:
    entries: tuple[DecisionDiaryEntry, ...]
    input_hash: str
    window: RIMWindow
    aggregates: Mapping[str, Any]


@dataclass(slots=True)
class StrategyStats:
    """Aggregated metrics per strategy for audit purposes."""

    entry_count: int
    mean_pnl: float
    pnl_std: float
    risk_rate: float
    win_rate: float
    loss_rate: float
    volatility_mean: float
    spread_mean: float
    belief_confidence_mean: float
    pnl_trend: float
    drawdown_ratio: float


@dataclass(slots=True)
class StrategyEncoding:
    strategy_id: str
    features: Mapping[str, float]
    stats: StrategyStats
    audit_entry_hashes: tuple[str, ...]


@dataclass(slots=True)
class StrategyInference:
    strategy_id: str
    weight_delta: float
    flag_probability: float
    experiment_probability: float
    confidence: float


@dataclass(slots=True)
class TRMSuggestion:
    suggestion_type: str
    payload: Mapping[str, Any]
    confidence: float
    rationale: str
    audit_ids: tuple[str, ...]
    suggestion_id: str


__all__ = [
    "DecisionDiaryEntry",
    "RIMInputBatch",
    "RIMWindow",
    "StrategyEncoding",
    "StrategyInference",
    "StrategyStats",
    "TRMSuggestion",
]
