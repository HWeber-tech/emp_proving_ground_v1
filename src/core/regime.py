"""
Core Regime Classification Port (Protocol)
=========================================

Defines a minimal, domain-agnostic interface for market regime detection so that
validation and other domain packages do not import concrete implementations.

Concrete adapters should live in higher layers (e.g., sensory or data providers)
and be injected at runtime by orchestration. This module provides a NoOp fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@dataclass
class RegimeResult:
    """Result structure returned by a RegimeClassifier implementation."""
    regime: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None


@runtime_checkable
class RegimeClassifier(Protocol):
    """
    Abstract regime classifier interface.

    Implementations should swallow internal errors and return a safe Optional[RegimeResult].
    """

    async def detect_regime(self, data: Any) -> Optional[RegimeResult]:
        """Analyze tabular/series-like market data and return a regime classification."""
        ...


class NoOpRegimeClassifier:
    """Safe default regime classifier that classifies nothing with zero confidence."""

    async def detect_regime(self, data: Any) -> Optional[RegimeResult]:
        try:
            return RegimeResult(regime="UNKNOWN", confidence=0.0, metadata={})
        except Exception:
            return RegimeResult(regime="UNKNOWN", confidence=0.0, metadata={})


def is_regime_classifier(obj: object) -> bool:
    """Runtime duck-typing helper."""
    try:
        return isinstance(obj, RegimeClassifier)
    except Exception:
        return False


__all__ = ["RegimeClassifier", "NoOpRegimeClassifier", "RegimeResult", "is_regime_classifier"]