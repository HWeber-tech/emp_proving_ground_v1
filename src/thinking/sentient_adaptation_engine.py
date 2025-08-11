"""
Shim module for SentientAdaptationEngine

This module re-exports the canonical SentientAdaptationEngine to avoid duplicate
class definitions across the codebase. Existing imports that reference
'thinking.sentient_adaptation_engine' will continue to work.
"""

from src.intelligence.sentient_adaptation import (
    SentientAdaptationEngine as SentientAdaptationEngine,
)

__all__ = ["SentientAdaptationEngine"]
