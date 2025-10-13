"""Governance module for system configuration and management."""

from __future__ import annotations

from .adaptive_gate import AdaptiveGovernanceGate
from .promotion import PromotionFeatureFlags, PromotionResult, promote_manifest_to_registry

__all__ = [
    "AdaptiveGovernanceGate",
    "PromotionFeatureFlags",
    "PromotionResult",
    "promote_manifest_to_registry",
]
