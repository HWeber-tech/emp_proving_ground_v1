from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest import mock

import pandas as pd
from sqlalchemy import create_engine

import pytest

from src.core.event_bus import Event, EventBus
from src.data_foundation.persist.timescale import TimescaleIngestor, TimescaleMigrator
from src.operations.retention import (
    DataRetentionSnapshot,
    EventPublishError,
    RetentionComponentSnapshot,
    RetentionPolicy,
    RetentionStatus,
    evaluate_data_retention,
    format_data_retention_markdown,
    publish_data_retention,
)


def _make_daily_frame(symbol: str, start: datetime, days: int) -> pd.DataFrame:
    index = pd.date_range(start, periods=days, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "ts": index,
            "symbol": [symbol] * days,
            "open": [1.0] * days,
            "high": [1.1] * days,
            "low": [0.9] * days,
            "close": [1.0] * days,
            "adj_close": [1.0] * days,
            "volume": [1_000] * days,
        }
    )


def _make_intraday_frame(symbol: str, start: datetime, bars: int) -> pd.DataFrame:
    index = pd.date_range(start, periods=bars, freq="H", tz="UTC")
    return pd.DataFrame(
        {
            "ts": index,
            "symbol": [symbol] * bars,
            "price": [1.0] * bars,
            "size": [100] * bars,
            "exchange": ["TEST"] * bars,
            "conditions": ["@"] * bars,
        }
    )


def _make_macro_frame(start: datetime, events: int) -> pd.DataFrame:
    index = pd.date_range(start, periods=events, freq="7D", tz="UTC")
    return pd.DataFrame(
        {
            "ts": index,
            "event_name": [f"NFP-{i}" for i in range(events)],
            "calendar": ["NFP"] * events,
            "currency": ["USD"] * events,
            "importance": ["high"] * events,
            "actual": [1.0] * events,
            "forecast": [1.0] * events,
            "previous": [1.0] * events,
        }
    )


def _seed_timescale(engine, reference: datetime) -> None:
    migrator = TimescaleMigrator(engine)
    migrator.apply()
    ingestor = TimescaleIngestor(engine)

    ingestor.upsert_daily_bars(
        _make_daily_frame("EURUSD", reference - timedelta(days=365), 365)
    )
    ingestor.upsert_intraday_trades(
        _make_intraday_frame("EURUSD", reference - timedelta(days=30), 30 * 24)
    )
    ingestor.upsert_macro_events(
        _make_macro_frame(reference - timedelta(days=365), 52)
    )


def test_evaluate_data_retention_pass(tmp_path) -> None:
    reference = datetime(2024, 1, 1, tzinfo=UTC)
    engine = create_engine(f"sqlite:///{tmp_path}/retention_pass.db")
    _seed_timescale(engine, reference)

    policies = (
        RetentionPolicy(
            dimension="daily_bars",
            schema="market_data",
            table="daily_bars",
            target_days=365,
            minimum_days=300,
            cap_days=365,
        ),
        RetentionPolicy(
            dimension="intraday_trades",
            schema="market_data",
            table="intraday_trades",
            target_days=30,
            minimum_days=7,
            cap_days=30,
        ),
        RetentionPolicy(
            dimension="macro_events",
            schema="macro_data",
            table="events",
            target_days=365,
            minimum_days=180,
            cap_days=365,
        ),
    )

    snapshot = evaluate_data_retention(engine, policies, reference=reference)
    assert snapshot.status is RetentionStatus.ok
    assert all(component.status is RetentionStatus.ok for component in snapshot.components)
    markdown = format_data_retention_markdown(snapshot)
    assert "| daily_bars |" in markdown


def test_evaluate_data_retention_exceeds_cap(tmp_path) -> None:
    reference = datetime(2024, 1, 1, tzinfo=UTC)
    engine = create_engine(f"sqlite:///{tmp_path}/retention_cap.db")
    _seed_timescale(engine, reference)

    policies = (
        RetentionPolicy(
            dimension="daily_bars",
            schema="market_data",
            table="daily_bars",
            target_days=365,
            minimum_days=300,
            cap_days=360,
        ),
    )

    snapshot = evaluate_data_retention(engine, policies, reference=reference)
    assert snapshot.status is RetentionStatus.fail
    assert snapshot.components[0].status is RetentionStatus.fail
    assert "exceeds cap" in snapshot.components[0].summary


def test_evaluate_data_retention_warn_and_optional(tmp_path) -> None:
    reference = datetime(2024, 6, 1, tzinfo=UTC)
    engine = create_engine(f"sqlite:///{tmp_path}/retention_warn.db")
    TimescaleMigrator(engine).apply()
    ingestor = TimescaleIngestor(engine)
    ingestor.upsert_daily_bars(_make_daily_frame("GBPUSD", reference - timedelta(days=120), 120))

    policies = (
        RetentionPolicy(
            dimension="daily_bars",
            schema="market_data",
            table="daily_bars",
            target_days=180,
            minimum_days=90,
        ),
        RetentionPolicy(
            dimension="intraday_trades",
            schema="market_data",
            table="intraday_trades",
            target_days=30,
            minimum_days=7,
            optional=True,
        ),
    )

    snapshot = evaluate_data_retention(engine, policies, reference=reference)
    statuses = {component.name: component.status for component in snapshot.components}
    assert statuses["daily_bars"] is RetentionStatus.warn
    assert statuses["intraday_trades"] is RetentionStatus.warn
    assert snapshot.status is RetentionStatus.warn


def test_publish_data_retention_emits_event() -> None:
    snapshot = DataRetentionSnapshot(
        status=RetentionStatus.ok,
        generated_at=datetime.now(tz=UTC),
        components=(
            RetentionComponentSnapshot(
                name="daily_bars",
                status=RetentionStatus.ok,
                summary="Coverage 365d",
            ),
        ),
    )

    class _StubBus(EventBus):
        def __init__(self) -> None:
            super().__init__()
            self.events: list[Event] = []

        def is_running(self) -> bool:  # pragma: no cover - trivial
            return True

        def publish_from_sync(self, event: Event) -> int:  # pragma: no cover - trivial
            self.events.append(event)
            return 1

    bus = _StubBus()

    publish_data_retention(bus, snapshot)

    assert bus.events
    assert bus.events[0].type == "telemetry.data_backbone.retention"


def test_publish_data_retention_falls_back_to_global_bus(monkeypatch) -> None:
    snapshot = DataRetentionSnapshot(
        status=RetentionStatus.ok,
        generated_at=datetime.now(tz=UTC),
        components=(
            RetentionComponentSnapshot(
                name="daily_bars",
                status=RetentionStatus.ok,
                summary="Coverage 365d",
            ),
        ),
    )

    class _RuntimeBus(EventBus):
        def is_running(self) -> bool:  # pragma: no cover - trivial
            return True

        def publish_from_sync(self, event: Event) -> int:
            raise RuntimeError("runtime bus failure")

    class _GlobalBus:
        def __init__(self) -> None:
            self.events: list[tuple[str, object, str | None]] = []

        def publish_sync(
            self, topic: str, payload: object, *, source: str | None = None
        ) -> None:
            self.events.append((topic, payload, source))

    global_bus = _GlobalBus()
    monkeypatch.setattr("src.operations.retention.get_global_bus", lambda: global_bus)

    publish_data_retention(_RuntimeBus(), snapshot)

    assert global_bus.events
    topic, payload, source = global_bus.events[0]
    assert topic == "telemetry.data_backbone.retention"
    assert source == "data_retention"
    assert payload["status"] == "ok"


def test_publish_data_retention_raises_on_unexpected_runtime_error(monkeypatch) -> None:
    snapshot = DataRetentionSnapshot(
        status=RetentionStatus.ok,
        generated_at=datetime.now(tz=UTC),
        components=(),
    )

    class _RuntimeBus(EventBus):
        def is_running(self) -> bool:  # pragma: no cover - trivial
            return True

        def publish_from_sync(self, event: Event) -> int:
            raise ValueError("unexpected failure")

    global_bus = mock.Mock()
    monkeypatch.setattr("src.operations.retention.get_global_bus", lambda: global_bus)

    with pytest.raises(EventPublishError):
        publish_data_retention(_RuntimeBus(), snapshot)

    global_bus.publish_sync.assert_not_called()


def test_evaluate_data_retention_rejects_unsafe_identifiers(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path}/retention_invalid.db")
    TimescaleMigrator(engine).apply()

    policies = (
        RetentionPolicy(
            dimension="invalid",
            schema="market_data",
            table="daily_bars",
            target_days=1,
            minimum_days=1,
            timestamp_column="ts; DROP TABLE",
        ),
    )

    with pytest.raises(ValueError):
        evaluate_data_retention(engine, policies)
