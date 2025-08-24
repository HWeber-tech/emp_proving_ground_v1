"""
Compatibility shims for legacy market_intelligence.dimensions.* imports.

These wrappers map legacy engine names to the canonical sensory engines in:
- src.sensory.organs.dimensions.how_organ
- src.sensory.organs.dimensions.what_organ
- src.sensory.organs.dimensions.when_organ
- src.sensory.organs.dimensions.why_organ
- src.sensory.organs.dimensions.anomaly_dimension
"""

from __future__ import annotations

__all__ = [
    "EnhancedFundamentalIntelligenceEngine",
    "InstitutionalIntelligenceEngine",
    "TechnicalRealityEngine",
    "ChronalIntelligenceEngine",
    "AnomalyIntelligenceEngine",
]
