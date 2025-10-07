"""Compatibility wrapper that re-exports the competitive intelligence surface."""

from __future__ import annotations

from typing import Any

from src.thinking.competitive.competitive_intelligence_system import (
    AlgorithmFingerprinter,
    BehaviorAnalyzer,
    CompetitiveIntelligenceSystem,
    CounterStrategyDeveloper,
    MarketShareTracker,
)

__all__ = [
    "AlgorithmFingerprinter",
    "BehaviorAnalyzer",
    "CompetitiveIntelligenceSystem",
    "CounterStrategyDeveloper",
    "MarketShareTracker",
]

_REMOVED_SYMBOLS = {
    "StrategyInsightLegacy",
    "AlgorithmFingerprinterLegacy",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - error paths exercised in tests
    """Raise a helpful error when callers request removed shims."""

    if name in _REMOVED_SYMBOLS:
        raise AttributeError(
            f"{name} has been removed. Import canonical classes from "
            "src.thinking.competitive.competitive_intelligence_system instead."
        )
    raise AttributeError(name)


def __dir__() -> list[str]:  # pragma: no cover - exercised indirectly
    return sorted(set(__all__))

