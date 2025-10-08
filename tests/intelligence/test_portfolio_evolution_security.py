import importlib

import pytest


def test_portfolio_evolution_module_removed() -> None:
    with pytest.raises(ModuleNotFoundError, match="ecosystem"):
        importlib.import_module("src.intelligence.portfolio_evolution")
