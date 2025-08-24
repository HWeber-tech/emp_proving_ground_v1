"""Core components for the market intelligence system (legacy).

Prefer importing canonical dataclasses from
`src.sensory.signals` or pydantic models in `src.sensory.core.base` where needed.
"""

from __future__ import annotations

from src.core.base import (  # re-export
    DimensionalReading,
    InstrumentMeta,
    MarketData,
    MarketRegime,
)

__all__ = [
    "MarketData",
    "DimensionalReading",
    "MarketRegime",
    "InstrumentMeta",
]
