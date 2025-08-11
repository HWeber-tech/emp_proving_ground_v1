#!/usr/bin/env python3
"""
Shim for FAISSPatternMemory and MemoryEntry (thinking layer)

Canonical implementations reside at:
- src/sentient/memory/faiss_pattern_memory.FAISSPatternMemory
- src/sentient/memory/faiss_pattern_memory.MemoryEntry

This module re-exports the canonical classes to maintain backward compatibility
for imports that reference src.thinking.memory.faiss_memory.
"""

from __future__ import annotations

from src.sentient.memory.faiss_pattern_memory import (
    FAISSPatternMemory as FAISSPatternMemory,
    MemoryEntry as MemoryEntry,
)

__all__ = ["FAISSPatternMemory", "MemoryEntry"]
