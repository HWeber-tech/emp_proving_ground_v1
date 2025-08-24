from collections.abc import Mapping

"""
Stub module to satisfy optional import: src.trading.risk.market_regime_detector
Runtime-safe no-op implementation for validation flows.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Mapping


class MarketRegimeDetector:
    async def detect_regime(self, data: Mapping[str, object]) -> SimpleNamespace:
        """
        No-op stub: returns a neutral/unknown regime with zero confidence.
        Shape matches usages like `result.regime.value` and `result.confidence`.
        """
        return SimpleNamespace(regime=SimpleNamespace(value="UNKNOWN"), confidence=0.0)
