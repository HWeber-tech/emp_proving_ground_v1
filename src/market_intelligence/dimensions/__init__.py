"""
Compatibility shims for legacy market_intelligence.dimensions.* imports.

These wrappers map legacy engine names to the canonical sensory surfaces in:
- src.sensory.enhanced.how_dimension
- src.sensory.enhanced.what_dimension
- src.sensory.enhanced.when_dimension
- src.sensory.enhanced.why_dimension
- src.sensory.enhanced.anomaly_dimension

The older `src.sensory.organs.dimensions.*` modules have been retired as part
of the dead-code cleanup; callers should import the enhanced dimensions or the
new sensor modules directly.
"""

from __future__ import annotations

__all__ = [
    "EnhancedFundamentalIntelligenceEngine",
    "InstitutionalIntelligenceEngine",
    "TechnicalRealityEngine",
    "ChronalIntelligenceEngine",
    "AnomalyIntelligenceEngine",
]
