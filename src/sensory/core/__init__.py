"""Core components for the market intelligence system (legacy).

Prefer importing canonical dataclasses from
`src.sensory.signals` or pydantic models in `src.sensory.core.base` where needed.
"""

from src.core.base import DimensionalReading, InstrumentMeta, MarketData, MarketRegime  # re-export

__all__ = [
    "MarketData",
    "DimensionalReading",
    "MarketRegime",
    "InstrumentMeta",
]

