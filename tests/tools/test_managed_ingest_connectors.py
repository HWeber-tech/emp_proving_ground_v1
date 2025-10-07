import json

import pytest

from src.governance.system_config import DataBackboneMode, EmpTier, SystemConfig

from tools.operations import managed_ingest_connectors as mic


def _build_config(extras: dict[str, str] | None = None) -> SystemConfig:
    payload = {
        "TIMESCALE_URL": "sqlite:///:memory:",
        "TIMESCALE_SYMBOLS": "EURUSD",
        "REDIS_URL": "redis://localhost:6379/0",
        "KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
    }
    if extras:
        payload.update(extras)
    return SystemConfig(
        tier=EmpTier.tier_1,
        data_backbone_mode=DataBackboneMode.institutional,
        extras=payload,
    )


def _patch_config(monkeypatch: pytest.MonkeyPatch, config: SystemConfig) -> None:
    monkeypatch.setattr(mic, "_load_system_config", lambda _: config)


def test_managed_connectors_cli_reports_success(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    config = _build_config()
    _patch_config(monkeypatch, config)

    exit_code = mic.main(["--connectivity", "--format", "json"])
    assert exit_code == 0

    captured = capsys.readouterr().out.strip()
    assert captured
    report = json.loads(captured)

    assert report["should_run"] is True

    manifest = {entry["name"]: entry for entry in report["manifest"]}
    assert manifest["timescale"]["configured"] is True
    assert manifest["redis"]["configured"] is True
    assert manifest["kafka"]["configured"] is True

    connectivity = {entry["name"]: entry for entry in report["connectivity"]}
    assert connectivity["timescale"]["healthy"] is True
    assert "error" not in connectivity["timescale"]
    assert connectivity["redis"]["healthy"] is True
    assert "error" not in connectivity["redis"]


def test_managed_connectors_cli_reports_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _build_config()
    _patch_config(monkeypatch, config)

    def failing_engine(self, *_args, **_kwargs):
        raise OSError("timescale unreachable")

    monkeypatch.setattr(
        "src.data_foundation.persist.timescale.TimescaleConnectionSettings.create_engine",
        failing_engine,
    )

    exit_code = mic.main(["--connectivity", "--format", "json"])
    assert exit_code == 0

    report = json.loads(capsys.readouterr().out.strip())
    connectivity = {entry["name"]: entry for entry in report["connectivity"]}
    timescale_snapshot = connectivity["timescale"]
    assert timescale_snapshot["healthy"] is False
    assert "timescale unreachable" in timescale_snapshot["error"]
