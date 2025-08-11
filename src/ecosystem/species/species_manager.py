"""
Canonical SpeciesManager for the ecosystem domain.

This module re-exports the SpeciesManager used by ecosystem evolution so that
all imports can converge on a single canonical path.

Phase 1/2 approach: re-export from thinking implementation to avoid code moves.
Later phases may relocate the implementation fully under src/ecosystem/.
"""

from src.thinking.ecosystem.specialized_predator_evolution import (
    SpeciesManager as SpeciesManager,
)

__all__ = ["SpeciesManager"]