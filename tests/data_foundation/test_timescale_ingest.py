from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

import pandas as pd
from sqlalchemy import create_engine, text
import pytest

from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    IntradayTradeIngestPlan,
    MacroEventIngestPlan,
    TimescaleBackboneOrchestrator,
    TimescaleBackbonePlan,
    ingest_macro_events,
    ingest_yahoo_daily_bars,
    ingest_yahoo_intraday_trades,
)
from src.data_foundation.persist.timescale import (
    TimescaleConnectionSettings,
    TimescaleIngestResult,
    TimescaleIngestor,
    TimescaleMigrator,
    _table_name,
)
from src.data_foundation.persist.timescale_reader import TimescaleReader
from src.data_foundation.schemas import MacroEvent


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": datetime(2024, 1, 2, tzinfo=timezone.utc),
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.05,
                "adj_close": 1.04,
                "volume": 1000,
                "symbol": "EURUSD",
            },
            {
                "date": datetime(2024, 1, 3, tzinfo=timezone.utc),
                "open": 1.05,
                "high": 1.2,
                "low": 1.0,
                "close": 1.18,
                "adj_close": 1.17,
                "volume": 1200,
                "symbol": "EURUSD",
            },
        ]
    )


def _sample_intraday(symbol: str = "EURUSD") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc),
                "symbol": symbol,
                "price": 1.01,
                "size": 1000,
                "exchange": "TEST",
                "conditions": "SYNTH",
            },
            {
                "timestamp": datetime(2024, 1, 2, 12, 1, tzinfo=timezone.utc),
                "symbol": symbol,
                "price": 1.015,
                "size": 1500,
                "exchange": "TEST",
                "conditions": "SYNTH",
            },
        ]
    )


def _sample_macro_events() -> list[MacroEvent]:
    return [
        MacroEvent(
            timestamp=datetime(2024, 1, 2, 13, 30, tzinfo=timezone.utc),
            calendar="FOMC",
            event="Rate Decision",
            currency="USD",
            actual=5.25,
            forecast=5.25,
            previous=5.25,
            importance="high",
            source="fred",
        ),
        MacroEvent(
            timestamp=datetime(2024, 1, 3, 9, 0, tzinfo=timezone.utc),
            calendar="ECB",
            event="Press Conference",
            currency="EUR",
            actual=None,
            forecast=None,
            previous=None,
            importance="medium",
            source="fred",
        ),
    ]


def test_timescale_table_name_rejects_invalid_identifiers() -> None:
    with pytest.raises(ValueError, match="schema identifier"):
        _table_name("market-data", "daily_bars", "postgresql")
    with pytest.raises(ValueError, match="table identifier"):
        _table_name("market_data", "daily-bars", "postgresql")


def test_timescale_ingestor_rejects_invalid_table(tmp_path) -> None:
    engine = create_engine("sqlite:///:memory:")
    ingestor = TimescaleIngestor(engine)
    ingest_ts = datetime.now(tz=timezone.utc)
    record = {
        "ts": ingest_ts,
        "symbol": "EURUSD",
        "source": "yahoo",
        "ingested_at": ingest_ts,
    }

    with pytest.raises(ValueError, match="table identifier"):
        ingestor._ingest_records(  # type: ignore[attr-defined]
            records=[record],
            schema="market_data",
            table="daily-bars",
            key_columns=("symbol", "ts"),
            update_columns=("source", "ingested_at"),
            all_columns=("ts", "symbol", "source", "ingested_at"),
            dimension="daily_bars",
            entity_key="symbol",
            timestamp_key="ts",
            ingest_ts=ingest_ts,
            source="yahoo",
        )


def test_timescale_migrator_creates_tables_sqlite() -> None:
    engine = create_engine("sqlite:///:memory:")
    migrator = TimescaleMigrator(engine)
    migrator.apply()

    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"),
        ).fetchall()
    table_names = {row[0] for row in rows}
    assert "market_data_daily_bars" in table_names
    assert "market_data_intraday_trades" in table_names
    assert "macro_data_events" in table_names
    assert "telemetry_ingest_runs" in table_names


def test_timescale_settings_from_mapping_defaults() -> None:
    settings = TimescaleConnectionSettings.from_mapping({})
    # Falls back to sqlite so tests can run without Timescale present
    assert settings.url.endswith("timescale_sim.db")
    assert settings.application_name == "emp-timescale-ingest"


def test_timescale_ingestor_is_idempotent(tmp_path) -> None:
    db_path = tmp_path / "timescale.db"
    engine = create_engine(f"sqlite:///{db_path}")
    TimescaleMigrator(engine).apply()
    ingestor = TimescaleIngestor(engine)

    frame = _sample_frame()
    result1 = ingestor.upsert_daily_bars(frame)
    assert result1.rows_written == len(frame)
    assert result1.freshness_seconds is not None
    assert result1.source == "yahoo"

    # Replay with changed data to verify upsert behaviour
    replay = frame.copy()
    replay.loc[1, "volume"] = 1300
    result2 = ingestor.upsert_daily_bars(replay)
    assert result2.rows_written == len(frame)
    assert result2.source == "yahoo"

    with engine.connect() as conn:
        stored = conn.execute(
            text(
                "SELECT symbol, COUNT(*) AS cnt, SUM(volume) AS total_volume "
                "FROM market_data_daily_bars GROUP BY symbol"
            ),
        ).fetchall()

    assert stored == [("EURUSD", 2, 2300.0)]


def test_timescale_ingestor_intraday_upsert(tmp_path) -> None:
    db_path = tmp_path / "intraday.db"
    engine = create_engine(f"sqlite:///{db_path}")
    TimescaleMigrator(engine).apply()
    ingestor = TimescaleIngestor(engine)

    frame = _sample_intraday()
    result = ingestor.upsert_intraday_trades(frame)
    assert result.rows_written == len(frame)
    assert result.dimension == "intraday_trades"
    assert result.source == "yahoo"

    replay = frame.copy()
    replay.loc[1, "size"] = 1750
    ingestor.upsert_intraday_trades(replay)

    with engine.connect() as conn:
        stored = conn.execute(
            text(
                "SELECT COUNT(*) FROM market_data_intraday_trades WHERE symbol = :symbol",
            ),
            {"symbol": "EURUSD"},
        ).scalar_one()

    assert stored == 2


def test_timescale_ingestor_macro_events(tmp_path) -> None:
    db_path = tmp_path / "macro.db"
    engine = create_engine(f"sqlite:///{db_path}")
    TimescaleMigrator(engine).apply()
    ingestor = TimescaleIngestor(engine)

    events = _sample_macro_events()
    frame = pd.DataFrame([event.dict() for event in events])
    result = ingestor.upsert_macro_events(frame)
    assert result.rows_written == len(events)
    assert result.dimension == "macro_events"
    assert result.source == "fred"

    replay = frame.copy()
    replay.loc[0, "actual"] = 5.5
    ingestor.upsert_macro_events(replay)

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT event_name, COUNT(*) AS cnt FROM macro_data_events GROUP BY event_name",
            ),
        ).fetchall()

    assert rows == [("Press Conference", 1), ("Rate Decision", 1)]


class _RecordingPublisher:
    def __init__(self) -> None:
        self.calls: list[tuple[TimescaleIngestResult, Mapping[str, object] | None]] = []

    def publish(
        self,
        result: TimescaleIngestResult,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.calls.append((result, metadata))


def test_timescale_orchestrator_publishes_ingest_events(tmp_path) -> None:
    db_path = tmp_path / "orch.db"
    settings = TimescaleConnectionSettings.from_mapping({"TIMESCALEDB_URL": f"sqlite:///{db_path}"})
    publisher = _RecordingPublisher()
    orchestrator = TimescaleBackboneOrchestrator(settings, event_publisher=publisher)

    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=["EURUSD"], lookback_days=30, source="yahoo"),
        intraday=IntradayTradeIngestPlan(
            symbols=["EURUSD"], lookback_days=1, interval="1m", source="yahoo"
        ),
        macro=MacroEventIngestPlan(events=_sample_macro_events(), source="fred"),
    )

    results = orchestrator.run(
        plan=plan,
        fetch_daily=lambda symbols, days: _sample_frame(),
        fetch_intraday=lambda symbols, days, interval: _sample_intraday(symbols[0]),
    )

    assert set(results) == {"daily_bars", "intraday_trades", "macro_events"}
    assert [call[0].dimension for call in publisher.calls] == [
        "daily_bars",
        "intraday_trades",
        "macro_events",
    ]

    daily_meta = publisher.calls[0][1]
    assert daily_meta is not None
    assert daily_meta["plan"] == "daily_bars"
    assert daily_meta["requested_symbols"] == ["EURUSD"]
    assert daily_meta["fetched_rows"] == 2

    intraday_meta = publisher.calls[1][1]
    assert intraday_meta is not None
    assert intraday_meta["interval"] == "1m"
    assert intraday_meta["fetched_rows"] == 2

    macro_meta = publisher.calls[2][1]
    assert macro_meta is not None
    assert macro_meta["plan"] == "macro_events"
    assert macro_meta["provided_events"] == len(_sample_macro_events())
    assert macro_meta.get("frame_rows") == len(_sample_macro_events())


def test_timescale_orchestrator_handles_empty_plan() -> None:
    settings = TimescaleConnectionSettings.from_mapping({})
    orchestrator = TimescaleBackboneOrchestrator(settings)

    def _should_not_run(*_: object, **__: object) -> None:
        raise AssertionError("fetcher should not be invoked for an empty plan")

    plan = TimescaleBackbonePlan()

    results = orchestrator.run(
        plan=plan,
        fetch_daily=_should_not_run,  # type: ignore[arg-type]
        fetch_intraday=_should_not_run,  # type: ignore[arg-type]
        fetch_macro=_should_not_run,  # type: ignore[arg-type]
    )

    assert results == {}


def test_timescale_orchestrator_fetches_macro_window(tmp_path) -> None:
    db_path = tmp_path / "macro_window.db"
    settings = TimescaleConnectionSettings(url=f"sqlite:///{db_path}")
    publisher = _RecordingPublisher()
    orchestrator = TimescaleBackboneOrchestrator(settings, event_publisher=publisher)

    plan = TimescaleBackbonePlan(
        macro=MacroEventIngestPlan(start="2024-01-01", end="2024-01-02", source="fred"),
    )

    def _fetch_window(start: str, end: str) -> list[MacroEvent]:
        assert start == "2024-01-01"
        assert end == "2024-01-02"
        return _sample_macro_events()

    results = orchestrator.run(plan=plan, fetch_macro=_fetch_window)

    assert set(results) == {"macro_events"}
    macro_result = results["macro_events"]
    assert macro_result.rows_written == len(_sample_macro_events())
    assert macro_result.dimension == "macro_events"
    assert len(publisher.calls) == 1

    result, metadata = publisher.calls[0]
    assert result is macro_result
    assert metadata is not None
    assert metadata["plan"] == "macro_events"
    assert metadata["window"] == {"start": "2024-01-01", "end": "2024-01-02"}
    assert metadata["provided_events"] == 0
    assert metadata["fetched_via_window"] is True
    assert metadata["fetched_events"] == len(_sample_macro_events())
    assert metadata["frame_rows"] == len(_sample_macro_events())


def test_timescale_orchestrator_requires_intraday_fetcher() -> None:
    settings = TimescaleConnectionSettings(url="sqlite:///:memory:")
    orchestrator = TimescaleBackboneOrchestrator(settings)

    plan = TimescaleBackbonePlan(
        intraday=IntradayTradeIngestPlan(symbols=["EURUSD"], lookback_days=1)
    )

    with pytest.raises(ValueError, match="Intraday ingest requested"):
        orchestrator.run(plan=plan, fetch_intraday=None)


def test_timescale_orchestrator_handles_macro_without_events(tmp_path) -> None:
    settings = TimescaleConnectionSettings(url="sqlite:///:memory:")
    publisher = _RecordingPublisher()
    orchestrator = TimescaleBackboneOrchestrator(settings, event_publisher=publisher)

    plan = TimescaleBackbonePlan(macro=MacroEventIngestPlan(source="fred"))

    results = orchestrator.run(plan=plan)

    assert set(results) == {"macro_events"}
    macro_result = results["macro_events"]
    assert macro_result.rows_written == 0
    assert macro_result.dimension == "macro_events"
    assert macro_result.source == "fred"

    assert len(publisher.calls) == 1
    _, metadata = publisher.calls[0]
    assert metadata is not None
    assert metadata["plan"] == "macro_events"
    assert metadata["window"] is None
    assert metadata["provided_events"] == 0


def test_ingest_pipeline_runs_with_sqlite(monkeypatch, tmp_path) -> None:
    captured: dict[str, pd.DataFrame] = {}

    def _fake_fetch(symbols: list[str], days: int = 60) -> pd.DataFrame:  # noqa: ARG001
        frame = _sample_frame()
        frame["symbol"] = symbols[0]
        captured["frame"] = frame
        return frame

    monkeypatch.setattr(
        "src.data_foundation.ingest.timescale_pipeline.fetch_daily_bars",
        _fake_fetch,
    )

    db_path = tmp_path / "pipeline.db"
    settings = TimescaleConnectionSettings(url=f"sqlite:///{db_path}")
    result = ingest_yahoo_daily_bars(settings=settings, symbols=["GBPUSD"], lookback_days=30)

    assert result.rows_written == len(captured["frame"])
    assert result.dimension == "daily_bars"
    engine = settings.create_engine()
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM market_data_daily_bars")).scalar_one()

    assert count == len(captured["frame"])


def test_ingest_intraday_pipeline(monkeypatch, tmp_path) -> None:
    captured: dict[str, pd.DataFrame] = {}

    def _fake_intraday(symbols: list[str], days: int, interval: str) -> pd.DataFrame:  # noqa: ARG001
        frame = _sample_intraday(symbols[0])
        captured["frame"] = frame
        return frame

    monkeypatch.setattr(
        "src.data_foundation.ingest.timescale_pipeline.fetch_intraday_trades",
        _fake_intraday,
    )

    db_path = tmp_path / "intraday_pipeline.db"
    settings = TimescaleConnectionSettings(url=f"sqlite:///{db_path}")
    result = ingest_yahoo_intraday_trades(settings=settings, symbols=["GBPUSD"], lookback_days=1)

    assert result.rows_written == len(captured["frame"])
    assert result.dimension == "intraday_trades"
    engine = settings.create_engine()
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM market_data_intraday_trades")).scalar_one()

    assert count == len(captured["frame"])


def test_ingest_macro_events_pipeline(tmp_path) -> None:
    events = _sample_macro_events()
    db_path = tmp_path / "macro_pipeline.db"
    settings = TimescaleConnectionSettings(url=f"sqlite:///{db_path}")
    result = ingest_macro_events(settings=settings, events=events)

    assert result.rows_written == len(events)
    assert result.dimension == "macro_events"
    engine = settings.create_engine()
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM macro_data_events")).scalar_one()

    assert count == len(events)


def test_backbone_orchestrator_runs_multi_plan(tmp_path) -> None:
    captured: dict[str, object] = {}

    def _fake_daily(symbols: list[str], days: int) -> pd.DataFrame:  # noqa: ARG001
        frame = _sample_frame()
        frame["symbol"] = symbols[0]
        captured["daily"] = frame
        return frame

    def _fake_intraday(symbols: list[str], days: int, interval: str) -> pd.DataFrame:  # noqa: ARG001
        frame = _sample_intraday(symbols[0])
        captured["intraday"] = frame
        return frame

    def _fake_macro(start: str, end: str) -> list[MacroEvent]:  # noqa: ARG001
        events = _sample_macro_events()
        captured["macro"] = events
        return events

    db_path = tmp_path / "plan.db"
    settings = TimescaleConnectionSettings(url=f"sqlite:///{db_path}")
    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=["EURUSD"], lookback_days=30),
        intraday=IntradayTradeIngestPlan(symbols=["EURUSD"], lookback_days=1),
        macro=MacroEventIngestPlan(start="2024-01-01", end="2024-01-05"),
    )
    orchestrator = TimescaleBackboneOrchestrator(settings)
    results = orchestrator.run(
        plan=plan,
        fetch_daily=_fake_daily,
        fetch_intraday=_fake_intraday,
        fetch_macro=_fake_macro,
    )

    assert set(results) == {"daily_bars", "intraday_trades", "macro_events"}
    assert results["daily_bars"].rows_written == len(captured["daily"])
    assert results["intraday_trades"].rows_written == len(captured["intraday"])
    assert results["macro_events"].rows_written == len(captured["macro"])


def test_timescale_reader_daily_filters(tmp_path) -> None:
    db_path = tmp_path / "reader_daily.db"
    engine = create_engine(f"sqlite:///{db_path}")
    TimescaleMigrator(engine).apply()
    ingestor = TimescaleIngestor(engine)
    frame = _sample_frame()
    ingestor.upsert_daily_bars(frame)

    reader = TimescaleReader(engine)
    result_all = reader.fetch_daily_bars(symbols=["EURUSD"])

    assert result_all.rowcount == len(frame)
    assert result_all.symbols == ("EURUSD",)
    assert result_all.dimension == "daily_bars"
    assert result_all.start_ts == frame["date"].min()
    assert result_all.end_ts == frame["date"].max()
    assert result_all.max_ingested_at is not None
    assert result_all.freshness_age_seconds(reference=datetime.now(timezone.utc)) >= 0

    start_filter = frame["date"].iloc[-1]
    result_window = reader.fetch_daily_bars(symbols=["EURUSD"], start=start_filter)
    assert result_window.rowcount == 1

    limited = reader.fetch_daily_bars(symbols=["EURUSD"], limit=1)
    assert limited.rowcount == 1

    latest = reader.latest_daily_bar("EURUSD")
    assert latest is not None
    assert latest["close"] == frame.iloc[-1]["close"]


def test_timescale_reader_intraday_slice(tmp_path) -> None:
    db_path = tmp_path / "reader_intraday.db"
    engine = create_engine(f"sqlite:///{db_path}")
    TimescaleMigrator(engine).apply()
    ingestor = TimescaleIngestor(engine)
    frame = _sample_intraday()
    ingestor.upsert_intraday_trades(frame)

    reader = TimescaleReader(engine)
    result = reader.fetch_intraday_trades(symbols=["EURUSD"], limit=1)

    assert result.dimension == "intraday_trades"
    assert result.rowcount == 1
    assert result.symbols == ("EURUSD",)
    assert result.max_ingested_at is not None
    assert set(result.frame.columns) >= {"ts", "symbol", "price", "size"}


def test_timescale_reader_macro_calendar_filter(tmp_path) -> None:
    db_path = tmp_path / "reader_macro.db"
    engine = create_engine(f"sqlite:///{db_path}")
    TimescaleMigrator(engine).apply()
    ingestor = TimescaleIngestor(engine)
    events = _sample_macro_events()
    df = pd.DataFrame([event.dict() for event in events])
    ingestor.upsert_macro_events(df)

    reader = TimescaleReader(engine)
    result = reader.fetch_macro_events(calendars=["FOMC"], start=events[0].timestamp)

    assert result.dimension == "macro_events"
    assert result.rowcount == 1
    assert result.symbols == ("FOMC",)
    assert result.frame.iloc[0]["event_name"] == "Rate Decision"
