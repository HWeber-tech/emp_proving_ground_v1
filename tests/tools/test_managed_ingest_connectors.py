from __future__ import annotations

import json

import pytest

from tools.operations.managed_ingest_connectors import main


@pytest.fixture(autouse=True)
def reset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure unrelated environment variables from the host do not leak into tests.
    monkeypatch.delenv("TIMESCALE_SYMBOLS", raising=False)
    monkeypatch.delenv("TIMESCALE_ENABLE_INTRADAY", raising=False)
    monkeypatch.delenv("TIMESCALEDB_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("KAFKA_BROKERS", raising=False)
    monkeypatch.delenv("KAFKA_INGEST_CONSUMER_TOPICS", raising=False)
    monkeypatch.delenv("DATA_BACKBONE_MODE", raising=False)
    monkeypatch.delenv("EMP_TIER", raising=False)


def test_cli_outputs_json(monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    timescale_db = tmp_path / "timescale.db"
    monkeypatch.setenv("DATA_BACKBONE_MODE", "institutional")
    monkeypatch.setenv("EMP_TIER", "tier_1")
    monkeypatch.setenv("TIMESCALE_SYMBOLS", "AAPL")
    monkeypatch.setenv("TIMESCALEDB_URL", f"sqlite:///{timescale_db}")
    monkeypatch.setenv("REDIS_URL", "redis://cache:6379/0")
    monkeypatch.setenv("KAFKA_BROKERS", "broker:9092")
    monkeypatch.setenv("KAFKA_INGEST_CONSUMER_TOPICS", "telemetry.ingest")

    exit_code = main(["--format", "json"])
    assert exit_code == 0

    captured = capsys.readouterr().out
    report = json.loads(captured)
    assert report["should_run"] is True
    assert report["dimensions"] == ["daily"]
    manifest = {entry["name"]: entry for entry in report["manifest"]}
    assert "telemetry.ingest" in manifest["kafka"]["metadata"]["topics"]
    assert manifest["redis"]["metadata"]["summary"].startswith("Redis")


def test_cli_markdown_with_connectivity(monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    timescale_db = tmp_path / "timescale-connectivity.db"
    monkeypatch.setenv("DATA_BACKBONE_MODE", "institutional")
    monkeypatch.setenv("EMP_TIER", "tier_1")
    monkeypatch.setenv("TIMESCALE_SYMBOLS", "AAPL")
    monkeypatch.setenv("TIMESCALEDB_URL", f"sqlite:///{timescale_db}")
    monkeypatch.setenv("KAFKA_BROKERS", "broker:9092")

    exit_code = main(["--format", "markdown", "--connectivity", "--timeout", "0.1"])
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "# Institutional Ingest Managed Connectors" in output
    assert "Managed Connectors" in output
    assert "Connectivity" in output
