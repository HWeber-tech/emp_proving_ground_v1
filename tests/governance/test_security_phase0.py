from __future__ import annotations

import importlib.util
import logging
import sys
import tempfile
from pathlib import Path

import pytest

import enum
import typing

if not hasattr(enum, "StrEnum"):
    class _CompatStrEnum(str, enum.Enum):
        """Minimal StrEnum backfill for Python < 3.11 used in tests."""

    enum.StrEnum = _CompatStrEnum  # type: ignore[attr-defined]

if not hasattr(typing, "Unpack"):
    typing.Unpack = typing.Any  # type: ignore[attr-defined]


_SRC_ROOT = Path(__file__).resolve().parents[2] / "src"


def _load_module(name: str, relative_path: str):
    module_path = _SRC_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SafetyManager = _load_module(
    "_test_security_phase0_safety_manager", "governance/safety_manager.py"
).SafetyManager
SystemConfigModule = _load_module(
    "_test_security_phase0_system_config", "governance/system_config.py"
)
SystemConfig = SystemConfigModule.SystemConfig


def test_enforce_raises_when_kill_switch_present(tmp_path: Path) -> None:
    kill_file = tmp_path / "emp_pg.KILL"
    kill_file.write_text("halt", encoding="utf-8")

    manager = SafetyManager(run_mode="paper", confirm_live=False, kill_switch_path=kill_file)

    with pytest.raises(RuntimeError):
        manager.enforce()


def test_enforce_logs_and_allows_when_kill_switch_path_unreadable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    manager = SafetyManager(run_mode="paper", confirm_live=False, kill_switch_path="guard/kill")
    kill_path = manager.kill_switch_path
    assert kill_path is not None

    path_type = type(kill_path)

    def _boom(self: Path) -> bool:
        raise OSError("fs error")

    monkeypatch.setattr(path_type, "exists", _boom, raising=True)

    with caplog.at_level(logging.WARNING):
        manager.enforce()

    assert any("Unable to inspect kill-switch path" in rec.message for rec in caplog.records)


def test_system_config_normalizes_kill_switch_from_env() -> None:
    cfg = SystemConfig.from_env({"EMP_KILL_SWITCH": "signals/kill"})
    expected_path = Path(tempfile.gettempdir()) / "signals/kill"
    assert cfg.kill_switch_path == expected_path
    env_vars = cfg.to_env()
    assert env_vars["EMP_KILL_SWITCH"] == str(expected_path)

    disabled_cfg = SystemConfig.from_env({"EMP_KILL_SWITCH": "disabled"})
    assert disabled_cfg.kill_switch_path is None
