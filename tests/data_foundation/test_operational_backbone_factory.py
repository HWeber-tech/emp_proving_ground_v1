from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.core.event_bus import EventBus
from src.data_foundation.cache.redis_cache import (
    InMemoryRedis,
    ManagedRedisCache,
    RedisCachePolicy,
)
from src.data_foundation.pipelines.operational_backbone import (
    OperationalIngestRequest,
    create_operational_backbone_pipeline,
)
from src.data_integration.real_data_integration import RealDataManager
from src.governance.system_config import (
    ConnectionProtocol,
    DataBackboneMode,
    EmpEnvironment,
    EmpTier,
    RunMode,
    SystemConfig,
)
from src.runtime.task_supervisor import TaskSupervisor
from src.sensory.real_sensory_organ import RealSensoryOrgan


try:  # pragma: no cover - optional dependency used when available
    import fakeredis
except Exception:  # pragma: no cover - fakeredis is optional in CI
    fakeredis = None  # type: ignore[assignment]


def _daily_frame(base: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": base - timedelta(days=1),
                "symbol": "EURUSD",
                "open": 1.10,
                "high": 1.12,
                "low": 1.08,
                "close": 1.11,
                "adj_close": 1.105,
                "volume": 1200,
            },
            {
                "date": base,
                "symbol": "EURUSD",
                "open": 1.11,
                "high": 1.13,
                "low": 1.09,
                "close": 1.125,
                "adj_close": 1.12,
                "volume": 1500,
            },
        ]
    )


def _intraday_frame(base: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": base - timedelta(minutes=2),
                "symbol": "EURUSD",
                "price": 1.121,
                "size": 640,
                "exchange": "TEST",
                "conditions": "SYNTH",
            },
            {
                "timestamp": base - timedelta(minutes=1),
                "symbol": "EURUSD",
                "price": 1.124,
                "size": 720,
                "exchange": "TEST",
                "conditions": "SYNTH",
            },
        ]
    )


def _managed_cache(policy: RedisCachePolicy | None = None) -> ManagedRedisCache:
    policy = policy or RedisCachePolicy.institutional_defaults()
    if fakeredis is not None:
        client = fakeredis.FakeRedis()
    else:
        client = InMemoryRedis()
    return ManagedRedisCache(client, policy)


@pytest.mark.asyncio()
async def test_create_operational_backbone_pipeline(tmp_path):
    db_path = tmp_path / "operational_backbone.db"
    config = SystemConfig(
        run_mode=RunMode.paper,
        environment=EmpEnvironment.demo,
        tier=EmpTier.tier_1,
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={
            "TIMESCALEDB_URL": f"sqlite:///{db_path}",
            "KAFKA_INGEST_ENABLE_STREAMING": "false",
        },
    )

    event_bus = EventBus()
    sensory = RealSensoryOrgan()

    pipeline = create_operational_backbone_pipeline(
        config,
        event_bus=event_bus,
        sensory_organ=sensory,
        event_topics=("telemetry.sensory.snapshot",),
        manager_kwargs={"managed_cache": _managed_cache()},
    )

    # Sanity check that the pipeline is wired with a RealDataManager instance.
    manager = getattr(pipeline, "_manager", None)
    assert isinstance(manager, RealDataManager)

    base = datetime(2024, 6, 1, tzinfo=timezone.utc)
    request = OperationalIngestRequest(
        symbols=("eurusd",),
        daily_lookback_days=2,
        intraday_lookback_days=1,
        intraday_interval="1m",
    )

    try:
        result = await pipeline.execute(
            request,
            fetch_daily=lambda symbols, lookback: _daily_frame(base),
            fetch_intraday=lambda symbols, lookback, interval: _intraday_frame(base),
            poll_consumer=False,
        )
    finally:
        await pipeline.shutdown()

    assert "daily_bars" in result.ingest_results
    assert result.ingest_results["daily_bars"].rows_written == 2
    assert "intraday_trades" in result.ingest_results
    assert result.ingest_results["intraday_trades"].rows_written == 2

    assert result.frames["daily_bars"].iloc[-1]["close"] == pytest.approx(1.125)
    assert result.frames["intraday_trades"].iloc[-1]["price"] == pytest.approx(1.124)

    assert result.sensory_snapshot is not None
    assert result.sensory_snapshot["symbol"] == "EURUSD"

    metrics_after = result.cache_metrics_after_fetch
    assert int(metrics_after.get("hits", 0)) >= 0

    assert result.task_snapshots
    assert any(
        entry.get("name") == "operational.backbone.ingest" for entry in result.task_snapshots
    )

    assert not event_bus.is_running()


@pytest.mark.asyncio()
async def test_operational_backbone_pipeline_uses_provided_supervisor(tmp_path) -> None:
    db_path = tmp_path / "operational_backbone_supervisor.db"
    config = SystemConfig(
        run_mode=RunMode.paper,
        environment=EmpEnvironment.demo,
        tier=EmpTier.tier_1,
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={
            "TIMESCALEDB_URL": f"sqlite:///{db_path}",
            "KAFKA_INGEST_ENABLE_STREAMING": "false",
        },
    )

    event_bus = EventBus()
    supervisor = TaskSupervisor(namespace="operational-pipeline-test")

    pipeline = create_operational_backbone_pipeline(
        config,
        event_bus=event_bus,
        sensory_organ=None,
        task_supervisor=supervisor,
        manager_kwargs={"managed_cache": _managed_cache()},
    )

    try:
        resolved_supervisor = getattr(pipeline, "_task_supervisor", None)
        owns_flag = getattr(pipeline, "_owns_task_supervisor", None)
        assert resolved_supervisor is supervisor
        assert owns_flag is False
    finally:
        await pipeline.shutdown()


def test_create_operational_backbone_pipeline_requires_timescale(tmp_path):
    config = SystemConfig(
        run_mode=RunMode.paper,
        environment=EmpEnvironment.demo,
        tier=EmpTier.tier_1,
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={"KAFKA_INGEST_ENABLE_STREAMING": "false"},
    )

    event_bus = EventBus()

    with pytest.raises(RuntimeError, match="Timescale connection required"):
        create_operational_backbone_pipeline(
            config,
            event_bus=event_bus,
            sensory_organ=None,
            manager_kwargs={"managed_cache": _managed_cache()},
        )


def test_create_operational_backbone_pipeline_requires_kafka_when_streaming(tmp_path):
    db_path = tmp_path / "operational_backbone_kafka.db"
    config = SystemConfig(
        run_mode=RunMode.paper,
        environment=EmpEnvironment.demo,
        tier=EmpTier.tier_1,
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={
            "TIMESCALEDB_URL": f"sqlite:///{db_path}",
            "KAFKA_INGEST_ENABLE_STREAMING": "true",
        },
    )

    event_bus = EventBus()

    with pytest.raises(RuntimeError, match="Kafka connection required"):
        create_operational_backbone_pipeline(
            config,
            event_bus=event_bus,
            sensory_organ=None,
            manager_kwargs={"managed_cache": _managed_cache()},
        )
