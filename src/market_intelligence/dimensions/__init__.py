"""Compatibility shims for legacy ``market_intelligence.dimensions`` imports.

The canonical sensory implementations live under ``src.sensory.enhanced``.  This
module lazily forwards attribute access so legacy import paths continue to work
without leaking the old namespace into new internal code.
"""

from __future__ import annotations

import importlib
from typing import Any, Final

__all__ = [
    "EnhancedFundamentalIntelligenceEngine",
    "InstitutionalIntelligenceEngine",
    "TechnicalRealityEngine",
    "ChronalIntelligenceEngine",
    "AnomalyIntelligenceEngine",
]

_LAZY_EXPORTS: Final[dict[str, str]] = {
    "EnhancedFundamentalIntelligenceEngine": (
        "src.sensory.enhanced.why_dimension:EnhancedFundamentalIntelligenceEngine"
    ),
    "InstitutionalIntelligenceEngine": (
        "src.sensory.enhanced.how_dimension:InstitutionalIntelligenceEngine"
    ),
    "TechnicalRealityEngine": (
        "src.sensory.enhanced.what_dimension:TechnicalRealityEngine"
    ),
    "ChronalIntelligenceEngine": (
        "src.sensory.enhanced.when_dimension:ChronalIntelligenceEngine"
    ),
    "AnomalyIntelligenceEngine": (
        "src.sensory.enhanced.anomaly_dimension:AnomalyIntelligenceEngine"
    ),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module, _, attr = target.partition(":")
    resolved = getattr(importlib.import_module(module), attr)
    globals()[name] = resolved  # cache for subsequent lookups
    return resolved


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
