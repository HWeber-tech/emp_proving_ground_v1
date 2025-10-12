"""Legacy compatibility shim for the sentient adaptation engine.

This module previously hosted the implementation of ``SentientAdaptationEngine``
under the historical ``src.intelligence`` namespace.  The canonical
implementation now lives in ``src.sentient.adaptation.sentient_adaptation_engine``.
For backwards compatibility we re-export the public API so existing imports
continue to function while the rest of the codebase migrates to the new
namespace.
"""

from __future__ import annotations

from src.sentient.adaptation.sentient_adaptation_engine import (
    AdaptationSignal,
    EpisodicMemorySystem,
    LearningSignal,
    MarketEvent,
    MetaCognitionEngine,
    MetaCognitionEngineImpl,
    SentientAdaptationEngine,
    logger,
    test_sentient_adaptation,
)

__all__ = [
    "AdaptationSignal",
    "EpisodicMemorySystem",
    "LearningSignal",
    "MarketEvent",
    "MetaCognitionEngine",
    "MetaCognitionEngineImpl",
    "SentientAdaptationEngine",
    "logger",
    "test_sentient_adaptation",
]
