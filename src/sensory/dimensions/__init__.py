"""Compatibility helpers for legacy sensory dimension imports.

The legacy ``src.sensory.organs.dimensions`` modules shipped placeholder
implementations that have now been removed as part of the dead-code cleanup
roadmap work.  Only the canonical WHAT dimension remains wired through the
pattern engine.  Any remaining imports of the retired dimensions now raise a
guardrail ``ModuleNotFoundError`` that points callers at the enhanced
replacements so consumers can migrate deliberately instead of silently falling
back to stale logic.
"""

from __future__ import annotations

from typing import Mapping, NoReturn

__all__ = ["WhatDimension"]

_LEGACY_DIMENSION_HINTS: Mapping[str, str] = {
    "AnomalyDimension": "src.sensory.enhanced.anomaly_dimension.AnomalyUnderstandingEngine",
    "ChaosDimension": "retired; chaos telemetry ships via src.sensory.real_sensory_organ",
    "HowDimension": "src.sensory.enhanced.how_dimension.InstitutionalUnderstandingEngine",
    "WhenDimension": "src.sensory.enhanced.when_dimension.ChronalUnderstandingEngine",
    "WhyDimension": "src.sensory.enhanced.why_dimension.EnhancedFundamentalUnderstandingEngine",
}


def __getattr__(name: str) -> NoReturn:
    """Provide explicit guidance for imports of retired sensory dimensions."""

    if name in _LEGACY_DIMENSION_HINTS:
        hint = _LEGACY_DIMENSION_HINTS[name]
        message = f"Legacy sensory dimension '{name}' has been removed. "
        if hint.startswith("retired;"):
            message += hint
        else:
            message += f"Import the enhanced implementation from '{hint}'."
        raise ModuleNotFoundError(message)
    if name == "WhatDimension":
        from src.sensory.organs.dimensions.pattern_engine import WhatDimension as _WhatDimension

        globals()[name] = _WhatDimension
        return _WhatDimension
    raise AttributeError(f"module 'src.sensory.dimensions' has no attribute '{name}'")
