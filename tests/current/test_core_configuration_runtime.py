"""Regression coverage for the legacy Configuration helper.

The core configuration module predates the typed `SystemConfig` helper but is
still imported by a handful of integration paths.  These tests guard the
behaviour we rely on during roadmap cleanup: environment overrides, nested
get/set accessors, YAML round-tripping, and the module-level loader's global
state contract.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest


def _reload_configuration(monkeypatch: pytest.MonkeyPatch):
    """Return a freshly loaded `src.core.configuration` module."""

    monkeypatch.delenv("EMP_ENVIRONMENT", raising=False)
    monkeypatch.delenv("EMP_DEBUG", raising=False)
    import src.core.configuration as configuration

    return importlib.reload(configuration)


def test_configuration_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _reload_configuration(monkeypatch)

    monkeypatch.setenv("EMP_ENVIRONMENT", "production")
    monkeypatch.setenv("EMP_DEBUG", "true")

    cfg = module.Configuration()

    assert cfg.environment == "production"
    assert cfg.debug is True


def test_configuration_get_set_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _reload_configuration(monkeypatch)
    cfg = module.Configuration()

    cfg.set("trading.limits.max_position", 5)
    cfg.set("sensory.providers.primary", "fix-sim")

    assert cfg.get("trading.limits.max_position") == 5
    assert cfg.get("sensory.providers.primary") == "fix-sim"
    assert cfg.trading == {"limits": {"max_position": 5}}
    assert cfg.sensory == {"providers": {"primary": "fix-sim"}}


@pytest.mark.parametrize(
    "path, default, expected",
    [
        ("system_name", "fallback", "EMP"),
        ("nonexistent.path", None, None),
        ("sensory.providers.secondary", "none", "none"),
    ],
)
def test_configuration_get_default(
    monkeypatch: pytest.MonkeyPatch, path: str, default: Any, expected: Any
) -> None:
    module = _reload_configuration(monkeypatch)
    cfg = module.Configuration()
    cfg.set("sensory.providers.primary", "fix-sim")

    assert cfg.get(path, default) == expected


def test_configuration_yaml_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _reload_configuration(monkeypatch)
    cfg = module.Configuration(system_name="Custom", system_version="9.9.9", environment="stage")
    cfg.set("trading.execution.mode", "paper")

    config_path = tmp_path / "config.yaml"
    cfg.to_yaml(config_path)

    loaded = module.Configuration.from_yaml(config_path)

    assert loaded.system_name == "Custom"
    assert loaded.system_version == "9.9.9"
    assert loaded.environment == "stage"
    assert loaded.get("trading.execution.mode") == "paper"


def test_load_config_updates_global_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _reload_configuration(monkeypatch)

    config_body = """
system_name: Roadmap
system_version: 2.0.0
environment: qa
trading:
  execution:
    mode: simulation
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_body.strip(), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    loaded = module.load_config()

    assert loaded is module.get_config()
    assert loaded.system_name == "Roadmap"
    assert loaded.get("trading.execution.mode") == "simulation"


def test_load_config_missing_file_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _reload_configuration(monkeypatch)

    with pytest.raises(module.ConfigurationException):
        module.Configuration.from_yaml(Path("/does/not/exist.yaml"))
