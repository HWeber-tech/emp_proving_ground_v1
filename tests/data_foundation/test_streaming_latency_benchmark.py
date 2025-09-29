from datetime import UTC, datetime, timedelta

from typing import Callable

from src.data_foundation.streaming import (
    LatencyBenchmarkReport,
    LatencySample,
    StreamingLatencyBenchmark,
)


def _clock_factory(start: datetime) -> Callable[[], datetime]:
    current = start

    def _clock() -> datetime:
        nonlocal current
        value = current
        current = current + timedelta(milliseconds=100)
        return value

    return _clock


def test_latency_benchmark_records_and_summarises() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    benchmark = StreamingLatencyBenchmark(clock=_clock_factory(start))

    producer_ts = start - timedelta(milliseconds=50)
    benchmark.record(producer_ts, dimension="daily_bars")
    benchmark.record(producer_ts, dimension="daily_bars")
    benchmark.record(producer_ts, dimension="intraday_trades")

    report = benchmark.summarise()

    assert isinstance(report, LatencyBenchmarkReport)
    assert report.overall.count == 3
    assert len(report.per_dimension) == 2
    per_dimension = {summary.dimension: summary for summary in report.per_dimension}
    assert per_dimension["daily_bars"].count == 2
    assert per_dimension["intraday_trades"].count == 1


def test_latency_benchmark_extend_and_clear() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    benchmark = StreamingLatencyBenchmark(clock=_clock_factory(start))
    sample = LatencySample(
        dimension="daily_bars",
        producer_ts=start,
        consumer_ts=start + timedelta(milliseconds=10),
        latency_ms=10.0,
    )

    benchmark.extend([sample])
    assert benchmark.samples() == (sample,)

    report = benchmark.summarise()
    assert report.overall.count == 1
    assert report.overall.avg_ms == 10.0

    benchmark.clear()
    assert benchmark.samples() == ()
    empty_report = benchmark.summarise()
    assert empty_report.overall.count == 0
