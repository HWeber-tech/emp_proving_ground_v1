"""
Core Regime Classification Port (Protocol)
=========================================

Defines a minimal, domain-agnostic interface for market regime classification.

- Do not import sensory/trading here (core must remain dependency-free).
- Domain packages should depend on this Protocol and receive implementations via DI.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable


class MarketRegime(Enum):
    """Canonical market regime taxonomy (core, dependency-free).

    Includes minimal generic labels and aliases used across layers.
    """
    # Minimal generic labels
    UNKNOWN = "UNKNOWN"
    CRISIS = "CRISIS"
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"

    # Aliases used by thinking layer (map to canonical values where appropriate)
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    TRENDING_UP = "BULLISH"
    TRENDING_DOWN = "BEARISH"
    BULL_MARKET = "BULLISH"
    BEAR_MARKET = "BEARISH"
    CONSOLIDATION = "SIDEWAYS"
    SIDEWAYS_MARKET = "SIDEWAYS"


@dataclass
class RegimeResult:
    """Classification output: a regime label and confidence in [0.0, 1.0]."""
    regime: str
    confidence: float


@runtime_checkable
class RegimeClassifier(Protocol):
    """Abstract regime classifier."""

    async def detect_regime(self, data: Any) -> Optional[RegimeResult]:
        """Classify a regime from a tabular/time-series dataset."""
        ...


class NoOpRegimeClassifier:
    """Safe fallback implementation that reports unknown regime."""

    async def detect_regime(self, data: Any) -> Optional[RegimeResult]:
        return RegimeResult(regime="UNKNOWN", confidence=0.0)


def is_regime_classifier(obj: object) -> bool:
    """Runtime duck-typing helper."""
    try:
        return isinstance(obj, RegimeClassifier)
    except Exception:
        return False