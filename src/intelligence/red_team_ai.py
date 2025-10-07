"""Compatibility wrapper that re-exports the canonical red team AI module."""

from __future__ import annotations

from typing import Any

from src.thinking.adversarial.red_team_ai import (
    AttackGenerator,
    ExploitDeveloper,
    RedTeamAI,
    StrategyAnalyzer,
    WeaknessDetector,
)

__all__ = [
    "StrategyAnalyzer",
    "WeaknessDetector",
    "AttackGenerator",
    "ExploitDeveloper",
    "RedTeamAI",
]

_REMOVED_SYMBOLS = {
    "StrategyAnalyzerLegacy",
    "WeaknessDetectorLegacy",
    "AttackGeneratorLegacy",
    "ExploitDeveloperLegacy",
    "RedTeamAILegacy",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - error paths exercised in tests
    """Raise a helpful error when callers access removed legacy shims."""

    if name in _REMOVED_SYMBOLS:
        raise AttributeError(
            f"{name} has been removed. Import canonical classes from "
            "src.thinking.adversarial.red_team_ai instead."
        )
    raise AttributeError(name)


def __dir__() -> list[str]:  # pragma: no cover - exercised indirectly
    return sorted(set(__all__))

