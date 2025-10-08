import json
from pathlib import Path

import pytest

from src.governance.system_config import DataBackboneMode, EmpTier, SystemConfig

from tools.operations import institutional_ingest_readiness as readiness


def _build_config(extra_overrides: dict[str, str] | None = None) -> SystemConfig:
    extras = {
        "TIMESCALE_URL": "sqlite:///:memory:",
        "TIMESCALE_SYMBOLS": "EURUSD",
        "REDIS_URL": "redis://localhost:6379/0",
    }
    if extra_overrides:
        extras.update(extra_overrides)
    return SystemConfig(
        tier=EmpTier.tier_1,
        data_backbone_mode=DataBackboneMode.institutional,
        extras=extras,
    )


def _patch_config(monkeypatch: pytest.MonkeyPatch, config: SystemConfig) -> None:
    monkeypatch.setattr(readiness, "_load_managed_config", lambda _args: config)


def test_readiness_cli_reports_connectors(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    config = _build_config()
    _patch_config(monkeypatch, config)

    exit_code = readiness.main(["--format", "json"])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out.strip())
    connectors = payload["managed_connectors"]
    assert connectors["should_run"] is True
    manifest = {entry["name"]: entry for entry in connectors["manifest"]}
    assert manifest["timescale"]["configured"] is True
    assert manifest["redis"]["configured"] is True
    assert manifest["kafka"]["configured"] is False
    assert "failover_drill" not in payload


def test_readiness_cli_runs_failover_drill(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    extras = {
        "TIMESCALE_URL": "sqlite:///:memory:",
        "TIMESCALE_SYMBOLS": "EURUSD",
        "TIMESCALE_FAILOVER_DRILL": "1",
        "TIMESCALE_FAILOVER_DRILL_DIMENSIONS": "daily",
    }
    config = _build_config(extras)
    _patch_config(monkeypatch, config)

    results_path = tmp_path / "timescale_results.json"
    results_path.write_text(
        json.dumps(
            {
                "daily": {
                    "dimension": "daily",
                    "rows_written": 25,
                    "symbols": ["EURUSD"],
                    "start_ts": "2024-01-01T00:00:00+00:00",
                    "end_ts": "2024-01-01T23:59:00+00:00",
                    "ingest_duration_seconds": 1.2,
                    "freshness_seconds": 30.0,
                    "source": "ingest",
                }
            }
        ),
        encoding="utf-8",
    )

    exit_code = readiness.main(
        [
            "--ingest-results",
            str(results_path),
            "--format",
            "json",
        ]
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out.strip())
    snapshot = payload["failover_drill"]
    assert snapshot["scenario"] == "required_timescale_failover"
    metadata = snapshot["metadata"]
    assert metadata["requested_dimensions"] == ["daily"]
    assert metadata["managed_manifest"]
