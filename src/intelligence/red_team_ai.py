"""Public façade for EMP red-team intelligence components.

This module intentionally re-exports the canonical implementations from
``src.thinking.adversarial.red_team_ai`` so external callers retain the
stable import path ``src.intelligence.red_team_ai`` while internal code can
depend on the canonical module directly.
"""

from __future__ import annotations

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
