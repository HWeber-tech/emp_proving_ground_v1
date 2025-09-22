from datetime import UTC, datetime, timedelta

from src.data_foundation.ingest.metrics import summarise_ingest_metrics
from src.data_foundation.persist.timescale import TimescaleIngestResult


def _result(*, dimension: str, rows: int, freshness: float | None) -> TimescaleIngestResult:
    now = datetime.now(tz=UTC)
    return TimescaleIngestResult(
        rows_written=rows,
        symbols=("EURUSD",),
        start_ts=now - timedelta(days=1),
        end_ts=now - timedelta(minutes=5),
        ingest_duration_seconds=1.2,
        freshness_seconds=freshness,
        dimension=dimension,
        source="test",
    )


def test_summarise_ingest_metrics_from_mapping() -> None:
    results = {
        "intraday_trades": _result(dimension="intraday_trades", rows=25, freshness=300.0),
        "daily_bars": _result(dimension="daily_bars", rows=50, freshness=60.0),
    }

    snapshot = summarise_ingest_metrics(results)

    assert snapshot.total_rows() == 75
    assert snapshot.active_dimensions() == ("daily_bars", "intraday_trades")
    payload = snapshot.as_dict()
    assert payload["total_rows"] == 75
    assert {item["dimension"] for item in payload["dimensions"]} == {
        "daily_bars",
        "intraday_trades",
    }


def test_summarise_ingest_metrics_from_iterable() -> None:
    values = [
        _result(dimension="macro_events", rows=0, freshness=None),
        _result(dimension="daily_bars", rows=10, freshness=120.0),
    ]

    snapshot = summarise_ingest_metrics(values)

    assert snapshot.total_rows() == 10
    assert snapshot.active_dimensions() == ("daily_bars",)
    assert snapshot.dimensions[0].dimension == "macro_events"
    assert snapshot.dimensions[1].freshness_seconds == 120.0
