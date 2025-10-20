from datetime import UTC, datetime, timedelta

from src.data_foundation.monitoring.feed_anomaly import (
    FeedAnomalyConfig,
    FeedHealthStatus,
    Tick,
    analyse_feed,
)


def _ticks(
    start: datetime,
    count: int,
    *,
    interval_seconds: int = 60,
    price: float = 100.0,
) -> list[Tick]:
    return [
        Tick(timestamp=start + timedelta(seconds=interval_seconds * index), price=price + index)
        for index in range(count)
    ]


def test_analyse_feed_returns_ok_when_no_anomalies() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    ticks = _ticks(start, 10)

    report = analyse_feed("EURUSD", ticks, now=start + timedelta(minutes=9))

    assert report.status is FeedHealthStatus.ok
    assert report.gaps == ()
    assert report.false_ticks == ()
    assert report.sample_count == 10
    assert not report.stale


def test_analyse_feed_flags_large_gap() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    ticks = _ticks(start, 5)
    ticks.append(Tick(timestamp=start + timedelta(minutes=20), price=150.0))

    config = FeedAnomalyConfig(max_gap_seconds=120, fail_gap_multiplier=20)
    report = analyse_feed("EURUSD", ticks, config=config, now=start + timedelta(minutes=21))

    assert report.gaps
    assert report.status is FeedHealthStatus.warn
    assert report.max_gap_seconds and report.max_gap_seconds > 600
    assert any("feed gaps" in issue for issue in report.issues)


def test_analyse_feed_flags_false_tick_reversion() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    ticks = _ticks(start, 5)
    # Insert false spike that reverts on the next tick.
    ticks.insert(3, Tick(timestamp=start + timedelta(minutes=3, seconds=30), price=200.0, volume=0.0))
    config = FeedAnomalyConfig(price_jump_pct=20.0, min_samples_for_ok=3)

    report = analyse_feed("EURUSD", ticks, config=config, now=start + timedelta(minutes=5))

    assert report.false_ticks
    assert report.status is FeedHealthStatus.warn
    assert any("suspect ticks" in issue for issue in report.issues)


def test_analyse_feed_marks_stale_as_failure() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    ticks = _ticks(start, 6)

    config = FeedAnomalyConfig(stale_grace_seconds=60)
    report = analyse_feed("EURUSD", ticks, config=config, now=start + timedelta(minutes=15))

    assert report.stale
    assert report.status is FeedHealthStatus.fail
    assert any("stale" in issue for issue in report.issues)


def test_analyse_feed_drops_out_of_order_ticks() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    ticks = [
        Tick(timestamp=start, price=100.0, seqno=1),
        Tick(timestamp=start, price=100.5, seqno=2),
        Tick(timestamp=start - timedelta(seconds=30), price=99.0, seqno=0),
        Tick(timestamp=start + timedelta(seconds=60), price=101.0, seqno=1),
        Tick(timestamp=start + timedelta(seconds=60), price=101.5, seqno=0),
    ]

    report = analyse_feed("EURUSD", ticks, now=start + timedelta(minutes=5))

    assert report.sample_count == 3
    assert len(report.dropped_ticks) == 2
    reason_codes = {tick.reason_code for tick in report.dropped_ticks}
    assert reason_codes == {"out_of_order_timestamp", "out_of_order_seqno"}
    assert report.metadata.get("dropped_tick_count") == 2
    assert any("Dropped 2" in issue for issue in report.issues)
    assert report.status is FeedHealthStatus.warn
