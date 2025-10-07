from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from tools.operations.run_production_ingest import main

from src.data_foundation.persist.timescale import TimescaleIngestResult


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    keys = {
        "DATA_BACKBONE_MODE",
        "EMP_TIER",
        "TIMESCALE_SYMBOLS",
        "TIMESCALEDB_URL",
        "TIMESCALE_INGEST_SCHEDULE",
        "TIMESCALE_INGEST_INTERVAL_SECONDS",
        "TIMESCALE_LOOKBACK_DAYS",
        "REDIS_URL",
        "KAFKA_BROKERS",
        "KAFKA_INGEST_CONSUMER_TOPICS",
    }
    for key in keys:
        monkeypatch.delenv(key, raising=False)


def _stub_ingest_result() -> TimescaleIngestResult:
    now = datetime.now(tz=UTC)
    return TimescaleIngestResult(
        rows_written=42,
        symbols=("EURUSD", "GBPUSD"),
        start_ts=now - timedelta(days=1),
        end_ts=now,
        ingest_duration_seconds=0.12,
        freshness_seconds=3.5,
        dimension="daily_bars",
        source="stub",
    )


def _patch_orchestrator(monkeypatch: pytest.MonkeyPatch) -> list:
    calls: list = []

    class _StubOrchestrator:
        def __init__(self, settings, event_publisher=None) -> None:  # noqa: D401 - test stub
            self._publisher = event_publisher

        def run(self, *, plan):  # noqa: D401 - interface match
            calls.append(plan)
            result = _stub_ingest_result()
            if self._publisher is not None:
                self._publisher.publish(result, metadata={"stub": True})
            return {"daily_bars": result}

    monkeypatch.setattr(
        "src.data_foundation.ingest.production_slice.TimescaleBackboneOrchestrator",
        _StubOrchestrator,
    )
    return calls


def test_cli_once_outputs_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _patch_orchestrator(monkeypatch)

    timescale_db = tmp_path / "cli-once.db"
    monkeypatch.setenv("DATA_BACKBONE_MODE", "institutional")
    monkeypatch.setenv("EMP_TIER", "tier_1")
    monkeypatch.setenv("TIMESCALE_SYMBOLS", "EURUSD,GBPUSD")
    monkeypatch.setenv("TIMESCALEDB_URL", f"sqlite:///{timescale_db}")

    exit_code = main(["--mode", "once", "--format", "json"])

    assert exit_code == 0
    assert len(calls) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["should_run"] is True
    assert payload["last_results"]["daily_bars"]["rows_written"] == 42
    assert payload["last_results"]["daily_bars"]["source"] == "stub"


def test_cli_schedule_runs_bootstrap_and_stops(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _patch_orchestrator(monkeypatch)

    start_calls: list[int] = []
    stop_calls: list[int] = []

    from src.data_foundation.ingest.institutional_vertical import InstitutionalIngestServices

    original_start = InstitutionalIngestServices.start
    original_stop = InstitutionalIngestServices.stop

    def _wrapped_start(self) -> None:
        start_calls.append(1)
        original_start(self)

    async def _wrapped_stop(self) -> None:
        stop_calls.append(1)
        await original_stop(self)

    monkeypatch.setattr(
        InstitutionalIngestServices,
        "start",
        _wrapped_start,
    )
    monkeypatch.setattr(
        InstitutionalIngestServices,
        "stop",
        _wrapped_stop,
    )

    timescale_db = tmp_path / "cli-schedule.db"
    monkeypatch.setenv("DATA_BACKBONE_MODE", "institutional")
    monkeypatch.setenv("EMP_TIER", "tier_1")
    monkeypatch.setenv("TIMESCALE_SYMBOLS", "EURUSD")
    monkeypatch.setenv("TIMESCALEDB_URL", f"sqlite:///{timescale_db}")

    exit_code = main(
        [
            "--mode",
            "schedule",
            "--duration",
            "0.05",
            "--format",
            "none",
        ]
    )

    assert exit_code == 0
    assert start_calls
    assert stop_calls
    # Bootstrap run should still execute once before the scheduler takes over.
    assert len(calls) >= 1

