import importlib

import pytest


def test_domain_models_module_removed() -> None:
    with pytest.raises(ModuleNotFoundError, match="trading telemetry payloads"):
        importlib.import_module("src.domain.models")


def test_domain_package_guard_for_execution_report() -> None:
    import src.domain as domain

    with pytest.raises(ModuleNotFoundError, match="ExecutionReport"):
        _ = domain.ExecutionReport
