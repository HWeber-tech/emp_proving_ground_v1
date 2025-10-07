"""Public façade for EMP competitive intelligence components.

The module narrows the legacy shim surface down to stable re-exports of the
canonical implementations that live under
``src.thinking.competitive.competitive_intelligence_system``.  External
callers can continue importing from ``src.intelligence.competitive_intelligence``
while internal code should depend on the canonical module directly.
"""

from __future__ import annotations

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
