from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from src.data_foundation.persist.timescale import TimescaleIngestResult
from tools.operations.run_failover_drill import main


@pytest.fixture(autouse=True)
def reset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATA_BACKBONE_MODE", raising=False)
    monkeypatch.delenv("EMP_TIER", raising=False)
    monkeypatch.delenv("TIMESCALE_SYMBOLS", raising=False)
    monkeypatch.delenv("TIMESCALEDB_URL", raising=False)
    monkeypatch.delenv("TIMESCALE_FAILOVER_DRILL", raising=False)
    monkeypatch.delenv("TIMESCALE_FAILOVER_DRILL_DIMENSIONS", raising=False)
    monkeypatch.delenv("KAFKA_BROKERS", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)


def _write_results(path) -> str:
    result = TimescaleIngestResult(
        12,
        ("AAPL", "MSFT"),
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 2, tzinfo=UTC),
        42.5,
        3600.0,
        "daily_bars",
        "alpaca",
    )
    payload = {"daily_bars": result.as_dict()}
    results_path = path / "results.json"
    results_path.write_text(json.dumps(payload), encoding="utf-8")
    return str(results_path)


def _configure_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("DATA_BACKBONE_MODE", "institutional")
    monkeypatch.setenv("EMP_TIER", "tier_1")
    monkeypatch.setenv("TIMESCALE_SYMBOLS", "AAPL,MSFT")
    monkeypatch.setenv("TIMESCALEDB_URL", f"sqlite:///{tmp_path / 'timescale.db'}")
    monkeypatch.setenv("TIMESCALE_FAILOVER_DRILL", "true")
    monkeypatch.setenv("TIMESCALE_FAILOVER_DRILL_DIMENSIONS", "daily_bars")


def test_cli_outputs_json(monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    _configure_env(monkeypatch, tmp_path)
    results_path = _write_results(tmp_path)

    exit_code = main(["--results", results_path, "--format", "json"])
    assert exit_code == 0

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["scenario"] == "timescale_failover"
    assert payload["status"] in {"ok", "warn", "fail"}
    metadata = payload["metadata"]
    assert metadata["requested_dimensions"] == ["daily_bars"]
    assert any(component["name"] == "failover" for component in payload["components"])


def test_cli_outputs_markdown(monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    _configure_env(monkeypatch, tmp_path)
    results_path = _write_results(tmp_path)

    exit_code = main([
        "--results",
        results_path,
        "--format",
        "markdown",
        "--scenario",
        "drill-demo",
    ])
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "# Timescale Failover Drill (drill-demo)" in output
    assert "| Component | Status | Summary |" in output
    assert "Requested dimensions: daily_bars" in output


def test_cli_loads_env_file(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    env_path = tmp_path / "institutional.env"
    db_path = tmp_path / "timescale.db"
    env_path.write_text(
        "\n".join(
            [
                "DATA_BACKBONE_MODE=institutional",
                "EMP_TIER=tier_1",
                "TIMESCALE_SYMBOLS=AAPL,MSFT",
                f"TIMESCALEDB_URL=sqlite:///{db_path}",
                "TIMESCALE_FAILOVER_DRILL=true",
                "TIMESCALE_FAILOVER_DRILL_DIMENSIONS=daily_bars",
                "KAFKA_BROKERS=broker:9092",
                "REDIS_URL=redis://localhost:6379/0",
            ]
        ),
        encoding="utf-8",
    )

    results_path = _write_results(tmp_path)

    exit_code = main([
        "--env-file",
        str(env_path),
        "--results",
        results_path,
    ])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["scenario"] == "timescale_failover"
    assert payload["metadata"]["requested_dimensions"] == ["daily_bars"]
