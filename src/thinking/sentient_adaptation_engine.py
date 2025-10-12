"""Legacy SentientAdaptationEngine shim removed.

Callers must import ``SentientAdaptationEngine`` from the canonical
``src.sentient.adaptation.sentient_adaptation_engine`` module.
"""

from __future__ import annotations

raise ModuleNotFoundError(
    "src.thinking.sentient_adaptation_engine was removed. Import "
    "SentientAdaptationEngine from src.sentient.adaptation.sentient_adaptation_engine instead."
)
