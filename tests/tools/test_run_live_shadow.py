from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from src.core.event_bus import Event
from src.data_foundation.ingest.scheduler import IngestSchedulerState
from src.data_foundation.persist.timescale import TimescaleIngestResult
from src.data_foundation.pipelines.operational_backbone import OperationalBackboneResult
from src.data_integration.real_data_integration import (
    BackboneConnectivityReport,
    ConnectivityProbeSnapshot,
)
from src.governance.system_config import DataBackboneMode, SystemConfig

from tools.data_ingest import run_live_shadow as cli


@dataclass
class _DummyScheduler:
    state_snapshot: IngestSchedulerState

    def state(self) -> IngestSchedulerState:
        return self.state_snapshot


class _DummyConnectivity:
    def as_dict(self) -> dict[str, object]:
        return {
            "timescale": True,
            "redis": True,
            "kafka": True,
            "probes": [
                {"name": "timescale", "healthy": True, "status": "ok"},
                {"name": "redis", "healthy": True, "status": "ok"},
                {"name": "kafka", "healthy": True, "status": "ok"},
            ],
        }


class _DummyManager:
    def __init__(self) -> None:
        self.require_flags: dict[str, object] = {}
        self.scheduler_started = False
        self.scheduler_stopped = False
        self.scheduler_metadata: dict[str, object] | None = None
        self.cache_metrics_calls = 0
        self.shutdown_called = False
        self.scheduler_state = IngestSchedulerState(
            running=False,
            last_started_at=None,
            last_completed_at=None,
            last_success_at=None,
            consecutive_failures=0,
            next_run_at=None,
            interval_seconds=60.0,
            jitter_seconds=0.0,
            max_failures=3,
        )

    def start_ingest_scheduler(self, plan_factory, schedule, metadata=None):
        # Record that the scheduler started and capture metadata for assertions.
        self.scheduler_started = True
        self.scheduler_plan = plan_factory()
        self.scheduler_schedule = schedule
        self.scheduler_metadata = dict(metadata or {})
        return _DummyScheduler(self.scheduler_state)

    async def stop_ingest_scheduler(self) -> None:
        self.scheduler_stopped = True

    def cache_metrics(self, *, reset: bool = False) -> dict[str, int]:
        self.cache_metrics_calls += 1
        return {"hits": 1, "misses": 0}

    def connectivity_report(self) -> _DummyConnectivity:
        return _DummyConnectivity()

    async def shutdown(self) -> None:
        self.shutdown_called = True


class _DummyPipeline:
    def __init__(self, result) -> None:
        self._result = result
        self.executed_request: cli.OperationalIngestRequest | None = None
        self.streaming_started = False
        self.stream_metadata: dict[str, object] | None = None
        self.stop_streaming_called = False
        self.shutdown_called = False
        self.streaming_snapshots = {"EURUSD": {"symbol": "EURUSD", "confidence": 0.8}}

    async def execute(self, request, poll_consumer: bool = True):
        self.executed_request = request
        return self._result

    async def start_streaming(self, *, metadata=None):
        self.streaming_started = True
        self.stream_metadata = dict(metadata or {})
        return object()

    async def stop_streaming(self) -> None:
        self.stop_streaming_called = True
        self.streaming_started = False

    async def shutdown(self) -> None:
        self.shutdown_called = True


class _DummyService:
    def __init__(self, manager: _DummyManager, pipeline: _DummyPipeline) -> None:
        self.manager = manager
        self.pipeline = pipeline
        self.scheduler_handle: _DummyScheduler | None = None
        self.streaming_task = object()
        self.shutdown_called = False

    async def ingest_once(self, request, *, poll_consumer: bool = True):
        return await self.pipeline.execute(request, poll_consumer=poll_consumer)

    async def ensure_streaming(self, *, metadata=None, task_name=None):
        await self.pipeline.start_streaming(metadata=metadata)
        return self.streaming_task

    async def start_scheduler(self, plan_factory, schedule, *, metadata=None, task_supervisor=None):
        self.scheduler_handle = self.manager.start_ingest_scheduler(
            plan_factory,
            schedule,
            metadata=metadata,
        )
        return self.scheduler_handle

    async def stop_scheduler(self) -> None:
        await self.manager.stop_ingest_scheduler()
        self.scheduler_handle = None

    async def stop_streaming(self) -> None:
        await self.pipeline.stop_streaming()

    def cache_metrics(self, *, reset: bool = False) -> dict[str, int]:
        return self.manager.cache_metrics(reset=reset)

    def connectivity_report(self) -> _DummyConnectivity:
        return self.manager.connectivity_report()

    def streaming_snapshots(self) -> dict[str, dict[str, object]]:
        return dict(self.pipeline.streaming_snapshots)

    def task_snapshots(self):  # pragma: no cover - not exercised in stub
        return ()

    async def shutdown(self) -> None:
        await self.pipeline.shutdown()
        await self.manager.shutdown()
        self.shutdown_called = True


@pytest.fixture()
def _live_shadow_context(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    config = SystemConfig(
        data_backbone_mode=DataBackboneMode.institutional,
        extras={
            "TIMESCALE_URL": "sqlite:///:memory:",
            "REDIS_URL": "redis://localhost:6379/0",
            "KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
        },
    )

    frame = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            "symbol": ["EURUSD"],
            "close": [1.25],
        }
    )

    ingest_result = TimescaleIngestResult(
        rows_written=1,
        symbols=("EURUSD",),
        start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ingest_duration_seconds=0.1,
        freshness_seconds=5.0,
        dimension="daily_bars",
        source="fixture",
    )

    backbone_result = OperationalBackboneResult(
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
        sensory_snapshot={"symbol": "EURUSD", "generated_at": "2024-01-01T00:00:00Z"},
        connectivity_report=BackboneConnectivityReport(
            timescale=True,
            redis=False,
            kafka=True,
            probes=(
                ConnectivityProbeSnapshot(
                    name="timescale",
                    healthy=True,
                    status="ok",
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

    manager = _DummyManager()
    pipeline = _DummyPipeline(backbone_result)
    service = _DummyService(manager, pipeline)

    def _load_config(_args):
        return config

    def _build_manager(_config, **kwargs):
        manager.require_flags = dict(kwargs)
        return manager

    monkeypatch.setattr(cli, "_load_system_config", _load_config)
    monkeypatch.setattr(cli, "_build_manager", _build_manager)
    monkeypatch.setattr(cli, "_build_event_bus", lambda: object())
    monkeypatch.setattr(cli, "_build_pipeline", lambda **_: pipeline)
    monkeypatch.setattr(cli, "_build_service", lambda manager, pipeline: service)

    context = SimpleNamespace(
        config=config,
        manager=manager,
        pipeline=pipeline,
        service=service,
    )
    return context


def test_live_shadow_cli_json(
    _live_shadow_context: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(["--duration", "0", "--format", "json"])
    assert exit_code == 0

    output = capsys.readouterr().out.strip()
    payload = json.loads(output)
    assert payload["initial"]["ingest_results"]["daily_bars"]["rows_written"] == 1
    assert payload["scheduler"] is None
    assert payload["streaming"]["enabled"] is True
    assert "EURUSD" in payload["streaming"]["snapshots"]
    assert payload["connections"]["timescale_url"] == "sqlite:///:memory:"
    assert _live_shadow_context.pipeline.stream_metadata == {"origin": "live_shadow_cli"}
    assert _live_shadow_context.pipeline.streaming_started is False
    assert _live_shadow_context.pipeline.stop_streaming_called is True
    assert _live_shadow_context.pipeline.shutdown_called is True
    assert _live_shadow_context.manager.shutdown_called is True
    assert _live_shadow_context.service.shutdown_called is True


def test_live_shadow_cli_require_connectors(
    _live_shadow_context: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(["--duration", "0", "--format", "json", "--require-connectors"])
    assert exit_code == 0
    _ = capsys.readouterr()

    flags = _live_shadow_context.manager.require_flags
    assert flags.get("require_timescale") is True
    assert flags.get("require_redis") is True
    assert flags.get("require_kafka") is True


def test_live_shadow_cli_stream_toggle(
    _live_shadow_context: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(["--duration", "0", "--format", "json", "--no-stream"])
    assert exit_code == 0
    _ = capsys.readouterr()

    pipeline = _live_shadow_context.pipeline
    assert pipeline.streaming_started is False
    assert pipeline.stop_streaming_called is False
    assert pipeline.shutdown_called is True
