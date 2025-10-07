from __future__ import annotations

import importlib

import pytest


def test_core_risk_manager_shim_import_raises_helpful_error() -> None:
    with pytest.raises(ModuleNotFoundError) as excinfo:
        importlib.import_module("src.core.risk.manager")

    message = str(excinfo.value)
    assert "src.risk.manager" in message
    assert "RiskManager" in message


def test_trading_risk_management_shim_import_raises_helpful_error() -> None:
    with pytest.raises(ModuleNotFoundError) as excinfo:
        importlib.import_module("src.trading.risk_management")

    message = str(excinfo.value)
    assert "src.risk.manager" in message
    assert "RiskManager" in message
