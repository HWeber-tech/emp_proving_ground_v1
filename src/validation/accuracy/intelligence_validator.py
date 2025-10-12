"""Legacy alias for the understanding validator.

This module used to host the ``IntelligenceValidator`` implementation
before the terminology converged on the understanding loop.  Retain it as
an import-compatible thin shim that re-exports the canonical types while
explicitly pointing callers at the updated namespace.  This keeps existing
imports working without duplicating the actual implementation, avoiding the
namespace drift where two modules quietly evolve out of sync.
"""

from __future__ import annotations

from .understanding_validator import UnderstandingValidator, ValidationMetrics

# Preserve the historical class name for callers that still import from this
# module.  The alias points at the canonical implementation so behaviour stays
# consistent and future updates only need to touch the understanding module.
IntelligenceValidator = UnderstandingValidator

__all__ = ["ValidationMetrics", "IntelligenceValidator", "UnderstandingValidator"]

