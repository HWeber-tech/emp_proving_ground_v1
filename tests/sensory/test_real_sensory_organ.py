from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

import pandas as pd

from src.sensory.lineage_publisher import SensoryLineagePublisher
from src.sensory.real_sensory_organ import RealSensoryOrgan, SensoryDriftConfig


class _StubBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def is_running(self) -> bool:
        return True

    def publish_from_sync(self, event: Any) -> int:
        self.events.append((event.type, event.payload))
        return 1


class _FailingBus:
    def __init__(self) -> None:
        self.calls = 0

    def is_running(self) -> bool:
        return True

    def publish_from_sync(self, _event: Any) -> None:
        self.calls += 1
        raise RuntimeError("runtime bus failure")


class _RecordingTopicBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any, str]] = []

    def publish_sync(self, event_type: str, payload: Any, *, source: str) -> None:
        self.events.append((event_type, payload, source))


class _RecordingPublisher:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record(self, dimension: str, lineage: Any, **kwargs: Any) -> None:
        payload = {"dimension": dimension, "lineage": lineage}
        payload.update(kwargs)
        self.records.append(payload)


def _build_market_frame() -> pd.DataFrame:
    base = datetime.now(timezone.utc) - timedelta(minutes=40)
    rows: list[dict[str, Any]] = []
    price = 1.10
    for idx in range(40):
        price += 0.0004 if idx % 7 == 0 else 0.0001
        ts = base + timedelta(minutes=idx)
        rows.append(
            {
                "timestamp": ts,
                "symbol": "EURUSD",
                "open": price - 0.0002,
                "high": price + 0.0003,
                "low": price - 0.0003,
                "close": price,
                "volume": 1500 + idx * 25,
                "volatility": 0.0004 + 0.00005 * (idx % 5),
                "spread": 0.00005 + 0.000005 * idx,
                "depth": 5000 + idx * 150,
                "order_imbalance": -0.2 + idx * 0.01,
                "macro_bias": 0.1 + 0.01 * (idx % 3),
                "yield_curve": {"2Y": 0.02 + idx * 0.0001, "10Y": 0.03 + idx * 0.00005},
                "yield_2y": 0.02 + idx * 0.0001,
                "yield_10y": 0.03 + idx * 0.00005,
            }
        )
    return pd.DataFrame(rows)


def _build_order_book() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bid_price": [1.1050, 1.1045, 1.1040],
            "ask_price": [1.1055, 1.1060, 1.1065],
            "bid_size": [120000, 90000, 75000],
            "ask_size": [110000, 85000, 72000],
        }
    )


def test_real_sensory_organ_observe_builds_snapshot() -> None:
    frame = _build_market_frame()
    order_book = _build_order_book()
    macro_events = [frame["timestamp"].iloc[-1] + timedelta(minutes=30)]
    bus = _StubBus()

    organ = RealSensoryOrgan(event_bus=bus)
    snapshot = organ.observe(
        frame,
        order_book=order_book,
        macro_events=macro_events,
        metadata={"mode": "test"},
    )

    assert snapshot["symbol"] == "EURUSD"
    integrated = snapshot["integrated_signal"]
    assert integrated.strength != 0.0
    assert integrated.contributing == ["WHY", "WHAT", "WHEN", "HOW", "ANOMALY"]

    dimensions = snapshot["dimensions"]
    assert set(dimensions.keys()) == {"WHY", "WHAT", "WHEN", "HOW", "ANOMALY"}
    for payload in dimensions.values():
        assert "signal" in payload
        assert "confidence" in payload
        assert "metadata" in payload
        assert "quality" in payload
        assert isinstance(payload["quality"], Mapping)
        assert "source" in payload["quality"]
        assert "lineage" in payload
        assert isinstance(payload["lineage"], Mapping)

    how_metadata = dimensions["HOW"]["metadata"]
    assert "lineage" in how_metadata
    assert how_metadata["lineage"]["dimension"] == "HOW"
    how_quality = dimensions["HOW"]["quality"]
    assert how_quality.get("source") == "sensory.how"

    anomaly_value = dimensions["ANOMALY"]["value"]
    assert isinstance(anomaly_value, Mapping)
    assert "is_anomaly" in anomaly_value

    lineage = snapshot["lineage"].as_dict()
    assert lineage["dimension"] == "SENSORY_FUSION"
    assert lineage["inputs"]["HOW"]["signal"] == dimensions["HOW"]["signal"]

    audit_entries = organ.audit_trail(limit=3)
    assert len(audit_entries) == 1
    assert audit_entries[0]["dimensions"]["WHAT"]["confidence"] == dimensions["WHAT"]["confidence"]

    assert snapshot["drift_summary"] is None

    assert bus.events
    event_type, payload = bus.events[0]
    assert event_type == "telemetry.sensory.snapshot"
    assert payload["symbol"] == "EURUSD"
    assert "drift_summary" in payload
    assert payload["drift_summary"] is None
    metrics_payload = payload.get("metrics")
    assert isinstance(metrics_payload, Mapping)
    dimensions_metrics = metrics_payload.get("dimensions")
    assert isinstance(dimensions_metrics, Mapping)
    assert {"HOW", "ANOMALY"}.issubset(dimensions_metrics.keys())

    status = organ.status()
    latest = status["latest"]
    assert isinstance(latest, Mapping)
    how_summary = latest["dimensions"]["HOW"]
    assert "metadata" in how_summary
    assert "audit" in how_summary["metadata"]

    metrics = organ.metrics()
    how_metrics = metrics["dimensions"]["HOW"]
    telemetry = how_metrics.get("telemetry")
    assert telemetry is not None
    assert "audit_liquidity" in telemetry


def test_real_sensory_organ_handles_empty_frame() -> None:
    organ = RealSensoryOrgan()
    snapshot = organ.observe(None)

    assert snapshot["symbol"] == "UNKNOWN"
    integrated = snapshot["integrated_signal"]
    assert integrated.strength == 0.0
    assert organ.audit_trail(limit=1)


def test_real_sensory_organ_exposes_drift_summary_after_window() -> None:
    frame = _build_market_frame()
    bus = _StubBus()
    config = SensoryDriftConfig(
        baseline_window=3,
        evaluation_window=2,
        min_observations=1,
        z_threshold=1.5,
        sensors=("WHY", "HOW"),
    )
    organ = RealSensoryOrgan(event_bus=bus, drift_config=config)

    for _ in range(5):
        organ.observe(frame)

    status = organ.status()
    drift_summary = status["drift_summary"]
    assert drift_summary is not None
    assert drift_summary["parameters"]["baseline_window"] == 3
    assert drift_summary["parameters"]["evaluation_window"] == 2
    assert drift_summary["results"]

    latest_event = bus.events[-1][1]
    assert latest_event["drift_summary"] is not None
    assert latest_event["drift_summary"]["parameters"]["z_threshold"] == config.z_threshold


def test_real_sensory_organ_metrics_exposes_dimension_state() -> None:
    frame = _build_market_frame()
    organ = RealSensoryOrgan()
    organ.observe(frame)

    metrics = organ.metrics()

    assert metrics["samples"] >= 1
    assert metrics["integrated"] is not None
    dimensions = metrics["dimensions"]
    assert "HOW" in dimensions and "ANOMALY" in dimensions

    how_metrics = dimensions["HOW"]
    assert "signal" in how_metrics
    assert "threshold_state" in how_metrics
    assert "telemetry" in how_metrics


def test_real_sensory_organ_wires_lineage_publisher() -> None:
    frame = _build_market_frame()
    publisher = _RecordingPublisher()
    organ = RealSensoryOrgan(lineage_publisher=publisher)

    snapshot = organ.observe(frame)

    assert publisher.records, "expected lineage publisher to receive records"
    recorded_dimensions = {record["dimension"] for record in publisher.records}
    assert {"HOW", "ANOMALY"}.issubset(recorded_dimensions)

    for record in publisher.records:
        assert record["symbol"] == snapshot["symbol"]
        lineage = record["lineage"]
        if isinstance(lineage, dict):
            assert lineage.get("dimension") == record["dimension"]


def test_real_sensory_organ_status_exposes_lineage_history() -> None:
    frame = _build_market_frame()
    publisher = SensoryLineagePublisher(max_records=3)
    organ = RealSensoryOrgan(lineage_publisher=publisher)

    organ.observe(frame)

    status = organ.status()
    assert "lineage" in status
    history = status["lineage"]
    assert isinstance(history, list)
    assert history, "expected lineage history to contain recent records"
    first_entry = history[0]
    assert first_entry.get("dimension") in {"HOW", "ANOMALY"}

    latest = status.get("lineage_latest")
    assert isinstance(latest, dict)
    assert latest.get("dimension") == first_entry.get("dimension")


def test_real_sensory_organ_falls_back_to_global_bus() -> None:
    frame = _build_market_frame()
    failing_bus = _FailingBus()
    topic_bus = _RecordingTopicBus()

    organ = RealSensoryOrgan(
        event_bus=failing_bus,
        global_bus_factory=lambda: topic_bus,
    )

    organ.observe(frame)

    assert topic_bus.events, "expected global bus to capture sensory snapshot"
    event_type, payload, source = topic_bus.events[-1]
    assert event_type == "telemetry.sensory.snapshot"
    assert source == "sensory.real_organ"
    assert payload["symbol"] == "EURUSD"


def test_real_sensory_organ_fallback_signal_includes_quality_and_lineage() -> None:
    class _EmptySensor:
        def process(self, *_args, **_kwargs):  # type: ignore[override]
            return []

    organ = RealSensoryOrgan(what_sensor=_EmptySensor())

    snapshot = organ.observe(None)

    what_dimension = snapshot["dimensions"]["WHAT"]
    metadata = what_dimension["metadata"]
    quality = metadata["quality"]
    assert quality["source"] == "sensory.real_organ"
    assert quality["reason"] == "missing_signal"
    assert isinstance(quality["timestamp"], str)

    lineage = metadata["lineage"]
    assert lineage["dimension"] == "WHAT"
    assert lineage["metadata"]["reason"] == "missing_signal"

    value = what_dimension["value"]
    assert value["fallback"] is True
    assert value["reason"] == "missing_signal"
