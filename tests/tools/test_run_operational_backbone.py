from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.core.event_bus import Event
from src.data_foundation.persist.timescale import TimescaleIngestResult
from src.governance.system_config import DataBackboneMode, SystemConfig
from src.data_integration.real_data_integration import (
    BackboneConnectivityReport,
    ConnectivityProbeSnapshot,
)

from tools.data_ingest import run_operational_backbone as cli


@pytest.fixture()
def _patched_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SystemConfig(
        data_backbone_mode=DataBackboneMode.institutional,
        extras={
            "TIMESCALE_URL": "sqlite:///:memory:",
            "TIMESCALE_SYMBOLS": "EURUSD",
            "REDIS_URL": "redis://localhost:6379/0",
            "KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
        },
    )

    monkeypatch.setattr(cli, "_load_system_config", lambda args: config)
    monkeypatch.setattr(cli, "_build_manager", lambda _config, **_: object())
    monkeypatch.setattr(cli, "_build_event_bus", lambda: object())
    monkeypatch.setattr(cli, "_build_pipeline", lambda **kwargs: object())

    frame = pd.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 2, tzinfo=timezone.utc),
            ],
            "symbol": ["EURUSD", "EURUSD"],
            "close": [1.1, 1.2],
        }
    )

    ingest_result = TimescaleIngestResult(
        rows_written=2,
        symbols=("EURUSD",),
        start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_ts=datetime(2024, 1, 2, tzinfo=timezone.utc),
        ingest_duration_seconds=0.5,
        freshness_seconds=30.0,
        dimension="daily_bars",
        source="fixture",
    )

    result = cli.OperationalBackboneResult(
        ingest_results={"daily_bars": ingest_result},
        frames={"daily_bars": frame},
        kafka_events=(
            Event(
                type="telemetry.ingest",
                payload={"result": {"dimension": "daily_bars"}},
                source="test.kafka",
            ),
        ),
        cache_metrics_before={"hits": 0, "misses": 1},
        cache_metrics_after_ingest={"hits": 1, "misses": 1},
        cache_metrics_after_fetch={"hits": 2, "misses": 1},
        sensory_snapshot={
            "symbol": "EURUSD",
            "generated_at": "2024-01-02T00:00:00+00:00",
            "integrated_signal": {"confidence": 0.9, "strength": 0.5},
        },
        task_snapshots=(
            {
                "name": "ingest-loop",
                "state": "completed",
                "metadata": {"component": "timescale_ingest.scheduler"},
            },
        ),
        streaming_snapshots={
            "EURUSD": {
                "timestamp": "2024-01-02T00:00:00+00:00",
                "confidence": 0.9,
            }
        },
        connectivity_report=BackboneConnectivityReport(
            timescale=True,
            redis=False,
            kafka=True,
            probes=(
                ConnectivityProbeSnapshot(
                    name="timescale",
                    healthy=True,
                    status="ok",
                    latency_ms=12.5,
                    details={"backend": "sqlite"},
                ),
                ConnectivityProbeSnapshot(
                    name="redis",
                    healthy=False,
                    status="degraded",
                    error="in-memory",
                ),
                ConnectivityProbeSnapshot(
                    name="kafka",
                    healthy=True,
                    status="ok",
                ),
            ),
        ),
    )

    async def _fake_execute_pipeline(
        *, task_supervisor=None, **_: object
    ) -> cli.OperationalBackboneResult:
        if task_supervisor is not None:
            await task_supervisor.cancel_all()
        return result

    monkeypatch.setattr(cli, "_execute_pipeline", _fake_execute_pipeline)


def test_operational_backbone_cli_json(
    _patched_dependencies: None, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(["--format", "json"])
    assert exit_code == 0

    output = capsys.readouterr().out.strip()
    payload = json.loads(output)
    assert payload["symbols"] == ["EURUSD"]
    assert payload["connections"]["timescale_url"] == "sqlite:///:memory:"
    assert payload["frames"]["daily_bars"]["rows"] == 2
    assert payload["events"][0]["type"] == "telemetry.ingest"
    assert payload["task_supervision"][0]["name"] == "ingest-loop"
    assert "EURUSD" in payload["streaming_snapshots"]
    assert payload["connectivity_report"]["status"] == "degraded"


def test_operational_backbone_cli_markdown(
    _patched_dependencies: None, tmp_path: Path
) -> None:
    output_path = tmp_path / "summary.md"
    exit_code = cli.main(["--format", "markdown", "--output", str(output_path)])
    assert exit_code == 0

    text = output_path.read_text(encoding="utf-8")
    assert "Operational Backbone Summary" in text
    assert "daily_bars" in text
    assert "Supervised tasks: 1" in text


def test_operational_backbone_cli_require_connectors(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SystemConfig(
        data_backbone_mode=DataBackboneMode.institutional,
        extras={"TIMESCALE_URL": "sqlite:///:memory:"},
    )

    monkeypatch.setattr(cli, "_load_system_config", lambda args: config)

    captured: dict[str, object] = {}

    def _capture_manager(_config: SystemConfig, **kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(cli, "_build_manager", _capture_manager)
    monkeypatch.setattr(cli, "_build_event_bus", lambda: object())
    monkeypatch.setattr(cli, "_build_pipeline", lambda **_: object())

    async def _fake_execute_pipeline(
        *, task_supervisor=None, **_: object
    ) -> cli.OperationalBackboneResult:
        frame = pd.DataFrame({"timestamp": [], "symbol": []})
        if task_supervisor is not None:
            await task_supervisor.cancel_all()
        return cli.OperationalBackboneResult(
            ingest_results={},
            frames={},
            kafka_events=tuple(),
            cache_metrics_before={},
            cache_metrics_after_ingest={},
            cache_metrics_after_fetch={},
            sensory_snapshot=None,
            task_snapshots=tuple(),
        )

    monkeypatch.setattr(cli, "_execute_pipeline", _fake_execute_pipeline)

    exit_code = cli.main(["--require-connectors", "--format", "json"])
    assert exit_code == 0
    assert captured.get("require_timescale") is True
    assert captured.get("require_redis") is True
    assert captured.get("require_kafka") is True
