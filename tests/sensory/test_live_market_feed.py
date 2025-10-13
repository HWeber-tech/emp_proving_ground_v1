from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.data_foundation.monitoring.feed_anomaly import FeedHealthStatus
from src.data_integration.real_data_integration import (
    BackboneConnectivityReport,
    ConnectivityProbeSnapshot,
)
from src.sensory.real_sensory_organ import RealSensoryOrgan
from src.sensory.services.live_market_feed import LiveMarketFeedMonitor


def _market_frame(base: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": base - timedelta(minutes=2),
                "price": 1.101,
                "size": 100,
                "symbol": "EURUSD",
            },
            {
                "timestamp": base - timedelta(minutes=1),
                "price": 1.103,
                "size": 120,
                "symbol": "EURUSD",
            },
            {
                "timestamp": base,
                "price": 1.105,
                "size": 90,
                "symbol": "EURUSD",
            },
        ]
    )


class _SyncManager:
    def __init__(self, frame: pd.DataFrame, connectivity: BackboneConnectivityReport | None) -> None:
        self._frame = frame
        self._connectivity = connectivity
        self.fetch_calls = 0

    def fetch_data(self, *args, **kwargs):  # type: ignore[override]
        self.fetch_calls += 1
        return self._frame.copy()

    def connectivity_report(self) -> BackboneConnectivityReport | None:
        return self._connectivity


class _AsyncManager:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame
        self.async_calls = 0

    async def get_market_data(self, *args, **kwargs):  # type: ignore[override]
        self.async_calls += 1
        return self._frame.copy()


def test_capture_snapshot_builds_diagnostics() -> None:
    base = datetime.now(timezone.utc)
    frame = _market_frame(base)
    connectivity = BackboneConnectivityReport(
        timescale=True,
        redis=True,
        kafka=False,
        probes=(
            ConnectivityProbeSnapshot(
                name="timescale",
                healthy=True,
                status="ok",
                latency_ms=12.5,
            ),
        ),
    )

    manager = _SyncManager(frame, connectivity)
    monitor = LiveMarketFeedMonitor(manager=manager, organ=RealSensoryOrgan())

    snapshot = monitor.capture("EURUSD", interval="1m", include_connectivity=True)

    assert snapshot.symbol == "EURUSD"
    assert not snapshot.market_data.empty
    assert "close" in snapshot.market_data.columns
    assert snapshot.diagnostics.symbol == "EURUSD"
    assert snapshot.feed_report.sample_count == 3
    assert snapshot.feed_report.status in {FeedHealthStatus.ok, FeedHealthStatus.warn}
    assert snapshot.connectivity is connectivity

    payload = snapshot.as_dict()
    assert payload["symbol"] == "EURUSD"
    assert isinstance(payload["market_data"], list)
    assert payload["diagnostics"]["symbol"] == "EURUSD"
    assert payload["feed_report"]["sample_count"] == 3


@pytest.mark.asyncio()
async def test_capture_async_prefers_async_manager() -> None:
    base = datetime.now(timezone.utc)
    frame = _market_frame(base)
    manager = _AsyncManager(frame)
    monitor = LiveMarketFeedMonitor(manager=manager)

    snapshot = await monitor.capture_async("EURUSD")

    assert snapshot.symbol == "EURUSD"
    assert manager.async_calls == 1
    assert snapshot.connectivity is None


def test_capture_marks_stale_feed() -> None:
    base = datetime.now(timezone.utc) - timedelta(minutes=15)
    frame = _market_frame(base)
    manager = _SyncManager(frame, None)
    monitor = LiveMarketFeedMonitor(manager=manager)

    snapshot = monitor.capture("EURUSD")

    assert snapshot.feed_report.stale is True
    assert snapshot.feed_report.status is FeedHealthStatus.fail


def test_capture_raises_on_empty_frame() -> None:
    manager = _SyncManager(pd.DataFrame(), None)
    monitor = LiveMarketFeedMonitor(manager=manager)

    with pytest.raises(ValueError):
        monitor.capture("EURUSD")
