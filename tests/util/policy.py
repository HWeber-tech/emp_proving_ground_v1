"""Utilities for constructing policy ledger fixtures in tests."""

from __future__ import annotations


def promotion_checklist_metadata(
    *,
    oos_regime_grid: bool = True,
    leakage_checks: bool = True,
    risk_audit: bool = True,
) -> dict[str, dict[str, bool]]:
    """Return metadata payload marking promotion checklist items as satisfied."""

    return {
        "promotion_checklist": {
            "oos_regime_grid": bool(oos_regime_grid),
            "leakage_checks": bool(leakage_checks),
            "risk_audit": bool(risk_audit),
        }
    }


__all__ = ["promotion_checklist_metadata"]
