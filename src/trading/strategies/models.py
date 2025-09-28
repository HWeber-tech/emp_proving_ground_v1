"""Common dataclasses shared across roadmap strategy implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal

StrategyAction = Literal["BUY", "SELL", "FLAT"]

__all__ = ["StrategyAction", "StrategySignal"]


@dataclass(slots=True)
class StrategySignal:
    """Represents a normalised trading signal output by a strategy."""

    symbol: str
    action: StrategyAction
    confidence: float
    notional: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation of the signal."""

        return {
            "symbol": self.symbol,
            "action": self.action,
            "confidence": float(self.confidence),
            "notional": float(self.notional),
            "metadata": dict(self.metadata),
        }

    def is_active(self) -> bool:
        """Return ``True`` when the signal requests an actionable trade."""

        return self.action != "FLAT"
