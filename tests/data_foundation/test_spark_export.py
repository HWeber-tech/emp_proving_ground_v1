from datetime import UTC, datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from src.data_foundation.batch.spark_export import (
    SparkExportFormat,
    SparkExportJob,
    SparkExportPlan,
    SparkExportStatus,
    execute_spark_export_plan,
)
from src.data_foundation.persist.timescale import TimescaleIngestor, TimescaleMigrator
from src.data_foundation.persist.timescale_reader import TimescaleReader


def _seed_timescale(tmp_path):
    db_path = tmp_path / "spark_timescale.db"
    engine = create_engine(f"sqlite:///{db_path}")
    TimescaleMigrator(engine).apply()
    ingestor = TimescaleIngestor(engine)

    daily_frame = pd.DataFrame(
        [
            {
                "date": datetime(2024, 1, 2, tzinfo=timezone.utc),
                "open": 1.0,
                "high": 1.1,
                "low": 0.95,
                "close": 1.05,
                "adj_close": 1.04,
                "volume": 1000,
                "symbol": "EURUSD",
            },
            {
                "date": datetime(2024, 1, 3, tzinfo=timezone.utc),
                "open": 1.06,
                "high": 1.2,
                "low": 1.0,
                "close": 1.18,
                "adj_close": 1.17,
                "volume": 1200,
                "symbol": "EURUSD",
            },
        ]
    )
    ingestor.upsert_daily_bars(daily_frame)

    trades = pd.DataFrame(
        [
            {
                "timestamp": datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc),
                "symbol": "EURUSD",
                "price": 1.01,
                "size": 1500,
                "exchange": "TEST",
                "conditions": "SYNTH",
            },
            {
                "timestamp": datetime(2024, 1, 2, 12, 1, tzinfo=timezone.utc),
                "symbol": "EURUSD",
                "price": 1.015,
                "size": 1600,
                "exchange": "TEST",
                "conditions": "SYNTH",
            },
        ]
    )
    ingestor.upsert_intraday_trades(trades)

    return engine


def test_execute_spark_export_plan_writes_csv(tmp_path):
    engine = _seed_timescale(tmp_path)
    try:
        reader = TimescaleReader(engine)
        plan = SparkExportPlan(
            root_path=tmp_path / "exports",
            format=SparkExportFormat.csv,
            jobs=(
                SparkExportJob(
                    dimension="daily_bars",
                    symbols=("EURUSD",),
                    filename="daily.csv",
                ),
            ),
        )

        snapshot = execute_spark_export_plan(
            reader,
            plan,
            now=datetime(2024, 1, 4, tzinfo=UTC),
        )

        assert snapshot.status is SparkExportStatus.ok
        output = tmp_path / "exports" / "daily_bars" / "daily.csv"
        assert output.exists()
        exported = pd.read_csv(output)
        assert len(exported) == 2
        assert set(exported.columns) >= {"ts", "open", "close", "symbol"}
    finally:
        engine.dispose()


def test_execute_spark_export_plan_partitions_by_symbol(tmp_path):
    engine = _seed_timescale(tmp_path)
    try:
        reader = TimescaleReader(engine)
        plan = SparkExportPlan(
            root_path=tmp_path / "partitioned",
            format=SparkExportFormat.jsonl,
            partition_columns=("symbol",),
            jobs=(
                SparkExportJob(
                    dimension="intraday_trades",
                    symbols=("EURUSD",),
                    filename="intraday.jsonl",
                ),
            ),
        )

        snapshot = execute_spark_export_plan(reader, plan)
        assert snapshot.status is SparkExportStatus.ok
        job = snapshot.jobs[0]
        assert job.paths
        for path in job.paths:
            assert path.endswith("intraday.jsonl")
            assert "symbol=EURUSD" in path
            file_path = Path(path)
            assert file_path.exists()
            with file_path.open("r", encoding="utf-8") as handle:
                lines = handle.readlines()
            assert len(lines) == 2
    finally:
        engine.dispose()


def test_execute_spark_export_plan_marks_failures(tmp_path):
    engine = _seed_timescale(tmp_path)
    try:
        reader = TimescaleReader(engine)
        plan = SparkExportPlan(
            root_path=tmp_path / "failures",
            format=SparkExportFormat.csv,
            jobs=(SparkExportJob(dimension="unknown"),),
        )

        snapshot = execute_spark_export_plan(reader, plan)
        assert snapshot.status is SparkExportStatus.fail
        assert snapshot.jobs[0].status is SparkExportStatus.fail
        assert snapshot.jobs[0].issues
    finally:
        engine.dispose()
