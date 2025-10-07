from __future__ import annotations

import importlib

import pytest


def test_core_configuration_import_raises_helpful_error() -> None:
    with pytest.raises(ModuleNotFoundError) as excinfo:
        importlib.import_module("src.core.configuration")

    message = str(excinfo.value)
    assert "SystemConfig" in message
    assert "src.governance.system_config" in message
