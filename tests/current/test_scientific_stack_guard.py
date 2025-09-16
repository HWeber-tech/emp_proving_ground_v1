from __future__ import annotations

from types import SimpleNamespace
import importlib

import pytest

from src.system import requirements_check


def _format_version(parts: tuple[int, int, int]) -> str:
    return ".".join(str(component) for component in parts)


def test_check_scientific_stack_success(monkeypatch: pytest.MonkeyPatch) -> None:
    modules = {
        package: SimpleNamespace(__version__=_format_version(minimum))
        for package, minimum in requirements_check.MINIMUM_VERSIONS.items()
    }

    def fake_import(name: str):
        return modules[name]

    monkeypatch.setattr(importlib, "import_module", fake_import)

    versions = requirements_check.check_scientific_stack()
    assert versions == {key: value.__version__ for key, value in modules.items()}


def test_check_scientific_stack_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    packages = list(requirements_check.MINIMUM_VERSIONS)
    missing = packages[0]
    available = {
        package: SimpleNamespace(__version__=_format_version(requirements_check.MINIMUM_VERSIONS[package]))
        for package in packages
        if package != missing
    }

    def fake_import(name: str):
        if name == missing:
            raise ModuleNotFoundError(f"No module named '{missing}'")
        return available[name]

    monkeypatch.setattr(importlib, "import_module", fake_import)

    with pytest.raises(ImportError) as exc:
        requirements_check.check_scientific_stack()

    message = str(exc.value)
    assert missing in message
    assert "pip install -r requirements/base.txt" in message


def test_check_scientific_stack_outdated(monkeypatch: pytest.MonkeyPatch) -> None:
    modules: dict[str, SimpleNamespace] = {}
    for package, minimum in requirements_check.MINIMUM_VERSIONS.items():
        version_str = _format_version(minimum)
        if package == "pandas":
            version_str = "0.0.0"
        modules[package] = SimpleNamespace(__version__=version_str)

    def fake_import(name: str):
        return modules[name]

    monkeypatch.setattr(importlib, "import_module", fake_import)

    with pytest.raises(RuntimeError) as exc:
        requirements_check.check_scientific_stack()

    message = str(exc.value)
    assert "pandas" in message
    assert _format_version(requirements_check.MINIMUM_VERSIONS["pandas"]) in message
