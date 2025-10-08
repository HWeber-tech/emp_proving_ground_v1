"""Compatibility facade for canonical ``src.core.base`` primitives."""

from __future__ import annotations

import importlib
from typing import Any, Final

__all__ = [
    "MarketData",
    "DimensionalReading",
    "MarketRegime",
    "SensoryReading",
    "InstrumentMeta",
    "SensoryOrgan",
]

_TARGET: Final[str] = "src.core.base"

_MODULE: Any = importlib.import_module(_TARGET)


def _resolve(name: str) -> Any:
    resolved = getattr(_MODULE, name)
    globals()[name] = resolved
    return resolved


for _symbol in __all__:
    _resolve(_symbol)


del _symbol


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
