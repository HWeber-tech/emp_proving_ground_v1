import importlib

import pytest


def test_health_monitor_module_removed() -> None:
    with pytest.raises(ModuleNotFoundError, match="src.operations"):
        importlib.import_module("src.operational.health_monitor")
