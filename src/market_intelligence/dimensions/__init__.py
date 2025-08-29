"""
Compatibility shims for legacy market_intelligence.dimensions.* imports.

These wrappers map legacy engine names to the canonical sensory engines in:
- sensory.organs.dimensions.how_organ
- sensory.organs.dimensions.what_organ
- sensory.organs.dimensions.when_organ
- sensory.organs.dimensions.why_organ
- sensory.organs.dimensions.anomaly_dimension
"""

from __future__ import annotations

__all__ = [
    "EnhancedFundamentalIntelligenceEngine",
    "InstitutionalIntelligenceEngine",
    "TechnicalRealityEngine",
    "ChronalIntelligenceEngine",
    "AnomalyIntelligenceEngine",
]
