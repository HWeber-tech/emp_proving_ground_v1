"""Regression coverage for :mod:`src.governance.safety_manager`."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.governance.safety_manager import SafetyManager


def test_from_config_normalises_confirmation_flags() -> None:
    """String payloads should map to the correct confirmation posture."""

    config = {
        "run_mode": "live",
        "confirm_live": "false",
        "kill_switch_path": "kill.flag",
    }

    manager = SafetyManager.from_config(config)

    with pytest.raises(RuntimeError):
        manager.enforce()

    expected_parent = Path(tempfile.gettempdir()) / "kill.flag"
    assert manager.kill_switch_path == expected_parent


def test_from_config_rejects_unrecognised_confirmation_payload() -> None:
    """Unrecognised confirmation strings should fail fast."""

    with pytest.raises(ValueError):
        SafetyManager.from_config({"confirm_live": "maybe"})


def test_enforce_requires_live_confirmation(tmp_path: Path) -> None:
    """Live runs without confirmation are blocked."""

    manager = SafetyManager.from_config({"run_mode": "live", "confirm_live": False})

    with pytest.raises(RuntimeError):
        manager.enforce()

    kill_switch = tmp_path / "halt.flag"
    kill_switch.write_text("halt")
    manager = SafetyManager("paper", True, kill_switch)

    with pytest.raises(RuntimeError):
        manager.enforce()


def test_enforce_warns_when_kill_switch_unreadable(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """I/O errors while checking the kill-switch should not abort startup."""

    manager = SafetyManager("paper", True, Path("/unreadable.flag"))

    def _boom(self: Path) -> bool:  # pragma: no cover - exercised via patch
        raise OSError("denied")

    monkeypatch.setattr(Path, "exists", _boom)

    with caplog.at_level("WARNING"):
        manager.enforce()

    assert any("Unable to inspect kill-switch path" in record.message for record in caplog.records)
