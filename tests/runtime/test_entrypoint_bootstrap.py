"""Tests for deterministic boot helpers in the runtime entrypoint."""

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from src.governance.system_config import SystemConfig

from main import (
    DEFAULT_CONFIG_SNAPSHOT_PATH,
    _capture_configuration_snapshot,
    _resolve_config_snapshot_path,
)


class _StubLogger:
    def __init__(self) -> None:
        self.infos: list[str] = []

    def info(self, message: Any, *args: Any, **kwargs: Any) -> None:
        if args:
            message = message % args
        self.infos.append(str(message))

    def warning(self, *args: Any, **kwargs: Any) -> None:
        return None

    def debug(self, *args: Any, **kwargs: Any) -> None:
        return None

    def exception(self, *args: Any, **kwargs: Any) -> None:
        return None


def test_resolve_config_snapshot_path_defaults_when_missing() -> None:
    path = _resolve_config_snapshot_path({})

    assert path == DEFAULT_CONFIG_SNAPSHOT_PATH


def test_resolve_config_snapshot_path_allows_disabling() -> None:
    assert _resolve_config_snapshot_path({"CONFIG_SNAPSHOT_PATH": "off"}) is None


def test_capture_configuration_snapshot_persists(tmp_path) -> None:
    target = tmp_path / "snapshots" / "config_snapshot.json"
    extras = {"CONFIG_SNAPSHOT_PATH": str(target)}
    config = SystemConfig()

    persisted, error, attempted = _capture_configuration_snapshot(config, extras, rng_seed=99)

    assert error is None
    assert attempted == target
    assert persisted == target

    payload = json.loads(target.read_text(encoding="utf-8"))
    metadata = payload.get("metadata")
    assert metadata is not None
    assert metadata.get("rng_seed") == 99
    assert metadata.get("source") == "runtime_boot"


@pytest.mark.asyncio()
async def test_main_logs_live_config_diff(monkeypatch: pytest.MonkeyPatch) -> None:
    import main as entrypoint

    stub_logger = _StubLogger()
    monkeypatch.setattr(entrypoint, "logger", stub_logger)

    def _stub_from_env(cls):  # type: ignore[override]
        return SystemConfig().with_updated(
            extras={
                "LIVE_FLAG": "enabled",
                "CONFIG_SNAPSHOT_PATH": "off",
            }
        )

    monkeypatch.setattr(entrypoint.SystemConfig, "from_env", classmethod(_stub_from_env))
    monkeypatch.setattr(entrypoint, "configure_structlog", lambda **_: None)

    async def _failing_build_app(*, config: Any):
        raise RuntimeError("stop")

    monkeypatch.setattr(entrypoint, "build_professional_predator_app", _failing_build_app)

    with pytest.raises(RuntimeError):
        await entrypoint.main()

    diff_messages = [msg for msg in stub_logger.infos if "Live config diff" in msg]
    assert diff_messages, "expected live config diff log entry"
    assert "extras.LIVE_FLAG" in diff_messages[0]


@pytest.mark.asyncio()
async def test_main_applies_structlog_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import main as entrypoint

    stub_logger = _StubLogger()
    monkeypatch.setattr(entrypoint, "logger", stub_logger)

    def _stub_from_env(cls):  # type: ignore[override]
        return SystemConfig().with_updated(
            extras={
                "STRUCTLOG_LEVEL": "debug",
                "STRUCTLOG_OUTPUT_FORMAT": " keyvalue ",
                "STRUCTLOG_DESTINATION": " stdout ",
                "CONFIG_SNAPSHOT_PATH": "off",
            }
        )

    monkeypatch.setattr(entrypoint.SystemConfig, "from_env", classmethod(_stub_from_env))

    captured: dict[str, Any] = {}

    def _capture_configure_structlog(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(entrypoint, "configure_structlog", _capture_configure_structlog)

    async def _failing_build_app(*, config: Any):
        raise RuntimeError("stop")

    monkeypatch.setattr(entrypoint, "build_professional_predator_app", _failing_build_app)

    with pytest.raises(RuntimeError):
        await entrypoint.main()

    assert captured["level"] == logging.DEBUG
    assert captured["output_format"] == "keyvalue"
    assert captured["destination"] == "stdout"
