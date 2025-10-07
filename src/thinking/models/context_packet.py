"""Canonical context packet definitions for the thinking domain."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ContextPacket(BaseModel):
    """Primary thinking-layer event emitted by understanding pipelines."""

    timestamp: datetime
    symbol: str

    # Market state
    current_price: float
    current_cvd: float

    # CVD divergence analysis
    cvd_divergence: Literal["bullish", "bearish"] | None = None
    divergence_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    # Supporting telemetry
    price_history: list[float] = Field(default_factory=list)
    cvd_history: list[float] = Field(default_factory=list)

    # Metadata and provenance
    analysis_window: int = 20
    source: str = "CVDDivergenceDetector"
    metadata: dict[str, object] = Field(default_factory=dict)

    def __str__(self) -> str:
        divergence_str = f" ({self.cvd_divergence})" if self.cvd_divergence else ""
        return f"ContextPacket({self.symbol}{divergence_str} @ {self.timestamp})"

    @property
    def has_signal(self) -> bool:
        """Return ``True`` when divergence analysis produced a signal."""

        return self.cvd_divergence is not None

    @property
    def signal_strength(self) -> float:
        """Expose a float representation of the divergence confidence."""

        return self.divergence_confidence or 0.0


__all__ = ["ContextPacket"]
