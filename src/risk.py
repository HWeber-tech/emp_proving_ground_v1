"""
Risk Management Module (shim)

This module now re-exports the canonical RiskManager and RiskConfig while
retaining ValidationResult for backward compatibility. See
docs/reports/CANONICALIZATION_PLAN.md and docs/reports/MIGRATION_PLAN.md.
"""

from __future__ import annotations

from src.config.risk.risk_config import RiskConfig as RiskConfig  # canonical config

# Canonical re-exports
from src.core.risk.manager import RiskManager as RiskManager  # single source of truth

__all__ = ["RiskManager", "RiskConfig"]
