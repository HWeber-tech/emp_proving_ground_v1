from __future__ import annotations

from typing import Any, Dict, Optional


class AmbusherFitnessFunction:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Typed shim for ambusher fitness function accepting optional config."""
        self.config = dict(config or {})
        # no-op initialization; config kept for potential future use
        return None

    def score(self, x: float) -> float:
        """Compute a fitness score for the given input."""
        return 0.0


__all__ = ["AmbusherFitnessFunction"]
