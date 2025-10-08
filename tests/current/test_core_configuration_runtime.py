from __future__ import annotations

import importlib

import pytest


def test_core_configuration_module_removed() -> None:
    """Legacy core configuration shim must not be importable."""

    with pytest.raises(ModuleNotFoundError) as excinfo:
        importlib.import_module("src.core.configuration")

    assert "src.core.configuration" in str(excinfo.value)
