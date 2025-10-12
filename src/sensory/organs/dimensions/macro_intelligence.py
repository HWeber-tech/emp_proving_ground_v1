"""Legacy macro-intelligence shim removed.

The macro layer now lives under the enhanced sensory namespace. Import
``EnhancedFundamentalUnderstandingEngine`` from
``src.sensory.enhanced.why_dimension`` instead of relying on this retired
path.  The guardrail keeps namespace drift from reintroducing stale
intelligence terminology.
"""

from __future__ import annotations

raise ModuleNotFoundError(
    "src.sensory.organs.dimensions.macro_intelligence was removed. Import "
    "EnhancedFundamentalUnderstandingEngine from "
    "src.sensory.enhanced.why_dimension instead."
)
