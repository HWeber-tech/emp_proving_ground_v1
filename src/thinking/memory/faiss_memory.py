"""Legacy shim for FAISSPatternMemory removed in favour of canonical module.

Importers must migrate to ``src.sentient.memory.faiss_pattern_memory`` and use
the canonical ``FAISSPatternMemory`` and ``MemoryEntry`` definitions.
"""

from __future__ import annotations

raise ModuleNotFoundError(
    "src.thinking.memory.faiss_memory was removed. Import FAISSPatternMemory "
    "and MemoryEntry from src.sentient.memory.faiss_pattern_memory instead."
)

