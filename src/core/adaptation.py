"""
Core Adaptation Service Port (Protocol)
======================================

Defines a minimal, domain-agnostic interface for real-time adaptation so that
thinking and other domains do not import src.intelligence directly.

Concrete implementations should live in higher layers (e.g., src/intelligence)
and be injected at runtime by orchestration composition. This module provides
a NoOp fallback that is safe and never raises.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable
import logging

logger = logging.getLogger(__name__)


@runtime_checkable
class AdaptationService(Protocol):
    """Domain-agnostic real-time adaptation service."""

    async def initialize(self) -> bool:
        """Prepare the adaptation service for use. Returns True on success."""
        ...

    async def stop(self) -> bool:
        """Gracefully stop the adaptation service. Returns True on success."""
        ...

    async def adapt_in_real_time(
        self, market_event: Any, strategy_response: Any, outcome: Any
    ) -> Dict[str, Any]:
        """
        Process a market event and associated strategy outcome to adapt parameters.
        Must swallow errors and return a safe dict. Keys are not strictly required,
        but recommended keys include: 'success', 'quality', 'adaptations', 'confidence'.
        """
        ...


class NoOpAdaptationService:
    """Safe default adaptation service that performs no work and never raises."""

    def __init__(self) -> None:
        self._initialized: bool = False

    async def initialize(self) -> bool:
        try:
            self._initialized = True
            return True
        except Exception:
            return True

    async def stop(self) -> bool:
        try:
            self._initialized = False
            return True
        except Exception:
            return True

    async def adapt_in_real_time(
        self, market_event: Any, strategy_response: Any, outcome: Any
    ) -> Dict[str, Any]:
        try:
            return {
                "success": False,
                "quality": 0.0,
                "adaptations": [],
                "confidence": 0.0,
                "info": "noop",
            }
        except Exception:
            return {
                "success": False,
                "quality": 0.0,
                "adaptations": [],
                "confidence": 0.0,
                "info": "noop",
            }


def is_adaptation_service(obj: object) -> bool:
    """Runtime duck-typing helper."""
    try:
        return isinstance(obj, AdaptationService)
    except Exception:
        return False


__all__ = ["AdaptationService", "NoOpAdaptationService", "is_adaptation_service"]