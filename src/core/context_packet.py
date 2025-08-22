"""
Context Packet Event Model
Enhanced with CVD divergence analysis for the Thinking Layer
"""

from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class ContextPacket(BaseModel):
    """
    The Thinking Layer's primary output event.
    v4.1 Schema: Enhanced with CVD divergence analysis.
    """
    timestamp: datetime
    symbol: str
    
    # Market state
    current_price: float
    current_cvd: float
    
    # CVD Divergence Analysis
    cvd_divergence: Optional[Literal["bullish", "bearish"]] = None
    
    # Confidence metrics
    divergence_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Supporting data
    price_history: list[float] = Field(default_factory=list)
    cvd_history: list[float] = Field(default_factory=list)
    
    # Metadata
    analysis_window: int = 20  # Number of data points analyzed
    source: str = "CVDDivergenceDetector"
    
    # Additional context
    metadata: dict[str, object] = Field(default_factory=dict)
    
    def __str__(self):
        divergence_str = f" ({self.cvd_divergence})" if self.cvd_divergence else ""
        return f"ContextPacket({self.symbol}{divergence_str} @ {self.timestamp})"
    
    @property
    def has_signal(self) -> bool:
        """Check if this packet contains a trading signal."""
        return self.cvd_divergence is not None
    
    @property
    def signal_strength(self) -> float:
        """Calculate signal strength based on confidence."""
        return self.divergence_confidence or 0.0
