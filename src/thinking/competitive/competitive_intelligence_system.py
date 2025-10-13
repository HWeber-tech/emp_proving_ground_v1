"""Legacy shim removed for competitive understanding system.

This module previously exposed :class:`CompetitiveIntelligenceSystem`.  The
canonical implementation now lives in
``src.thinking.competitive.competitive_understanding_system`` under the
``CompetitiveUnderstandingSystem`` class.  Importing this legacy module raises
``ModuleNotFoundError`` to make the transition explicit.
"""

from __future__ import annotations

raise ModuleNotFoundError(
    "src.thinking.competitive.competitive_intelligence_system was removed. "
    "Import CompetitiveUnderstandingSystem from "
    "src.thinking.competitive.competitive_understanding_system instead."
)
