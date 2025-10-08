from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "src.core.risk.manager",
        "src.core.risk.position_sizing",
        "src.trading.risk_management",
    ],
)
def test_legacy_risk_modules_are_absent(module_name: str) -> None:
    """Ensure retired risk-layer shims remain unavailable."""

    with pytest.raises(ModuleNotFoundError) as excinfo:
        importlib.import_module(module_name)

    assert module_name in str(excinfo.value)
