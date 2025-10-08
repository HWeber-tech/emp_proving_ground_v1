"""Legacy SentientAdaptationEngine shim removed.

Callers must import ``SentientAdaptationEngine`` from the canonical
``src.intelligence.sentient_adaptation`` module.
"""

from __future__ import annotations

raise ModuleNotFoundError(
    "src.thinking.sentient_adaptation_engine was removed. Import "
    "SentientAdaptationEngine from src.intelligence.sentient_adaptation instead."
)

