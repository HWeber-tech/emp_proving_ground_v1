"""
Shim module for SentientAdaptationEngine (sensory/intelligence independent)

This module intentionally avoids importing from src.intelligence to satisfy
the domain-independence contract. Use orchestration to wire a concrete
AdaptationService implementation. This shim provides a minimal, non-raising
no-op engine to preserve import compatibility for legacy references.
"""

from __future__ import annotations


class SentientAdaptationEngine:
    """No-op shim. Wire a real AdaptationService via orchestration composition."""

    async def initialize(self) -> bool:
        return True

    async def stop(self) -> bool:
        return True

    async def adapt_in_real_time(self, *args: object, **kwargs: object) -> dict[str, object]:
        # Return a minimal dict to avoid downstream None handling
        return {"success": False, "engine": "noop", "details": "Use core AdaptationService via orchestration."}


__all__ = ["SentientAdaptationEngine"]
