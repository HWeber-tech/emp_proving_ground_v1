"""Legacy shim removed for the contextual fusion engine.

The contextual fusion engine now lives exclusively under the
``enhanced_understanding_engine`` namespace.  Importing this module used to
silently re-export the new implementation which encouraged namespace drift –
callers could continue depending on the deprecated intelligence surface
without noticing.  We now raise an explicit ``ModuleNotFoundError`` so any
remaining references fail fast with guidance on the canonical replacement.
"""

from __future__ import annotations

raise ModuleNotFoundError(
    "src.orchestration.enhanced_intelligence_engine was removed. "
    "Import ContextualFusionEngine and related helpers from "
    "src.orchestration.enhanced_understanding_engine instead."
)
