"""Legacy sensory re-exports for the market understanding system.

Prefer importing canonical dataclasses from ``src.core.base`` (re-exported
here) or the pydantic helpers in
``src.sensory.organs.dimensions.base_organ``. The historical "market
intelligence" terminology is maintained only for backwards compatibility with
older scripts.
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
