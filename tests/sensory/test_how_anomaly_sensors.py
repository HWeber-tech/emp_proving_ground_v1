from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.sensory.anomaly import AnomalySensor
from src.sensory.anomaly.anomaly_sensor import AnomalySensorConfig
from src.sensory.how.how_sensor import HowSensor
from src.sensory.lineage import SensorLineageRecord, SensorLineageRecorder


@dataclass(slots=True)
class _StubReading:
    signal_strength: float
    confidence: float
    context: dict[str, object]
    regime: object | None = None


class _StubAdapter(dict):
    def __init__(self, reading: _StubReading, **extras: float) -> None:
        super().__init__(extras)
        self.reading = reading


class _StubHowEngine:
    def __init__(self, *, strength: float, confidence: float, **extras: float) -> None:
        self._strength = strength
        self._confidence = confidence
        self._extras = extras

    def analyze_institutional_intelligence(self, payload):  # type: ignore[override]
        return _StubAdapter(
            _StubReading(
                signal_strength=self._strength,
                confidence=self._confidence,
                context={"mode": "test"},
            ),
            **self._extras,
        )


class _StubAnomalyEngine:
    def __init__(self, *, strength: float, confidence: float, **extras: float) -> None:
        self._strength = strength
        self._confidence = confidence
        self._extras = extras

    def analyze_anomaly_intelligence(self, payload):  # type: ignore[override]
        return _StubAdapter(
            _StubReading(
                signal_strength=self._strength,
                confidence=self._confidence,
                context={"mode": "test"},
            ),
            **self._extras,
        )


class _RecordingAnomalyEngine:
    def __init__(self) -> None:
        self.seen_payload: object | None = None

    def analyze_anomaly_intelligence(self, payload):  # type: ignore[override]
        self.seen_payload = payload
        latest = payload[-1] if isinstance(payload, list) and payload else 0.0
        return _StubAdapter(
            _StubReading(
                signal_strength=0.52,
                confidence=0.05,
                context={"mode": "sequence"},
            ),
            baseline=0.4,
            dispersion=0.12,
            latest=latest,
        )


def _build_market_frame(rows: int = 12, *, anomaly_spike: bool = False) -> pd.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    data: list[dict[str, object]] = []
    price = 1.10
    for idx in range(rows):
        price += 0.0005 if idx % 2 == 0 else -0.0003
        if anomaly_spike and idx == rows - 1:
            price += 0.01
        data.append(
            {
                "timestamp": base + timedelta(minutes=idx),
                "symbol": "EURUSD",
                "open": price - 0.0004,
                "high": price + 0.0006,
                "low": price - 0.0005,
                "close": price,
                "volume": 1500 + idx * 120,
                "volatility": 0.0004 + idx * 0.00001,
                "spread": 0.00005,
                "depth": 5500 + idx * 120,
                "order_imbalance": 0.15 + 0.01 * idx,
                "data_quality": 0.9,
            }
        )
    return pd.DataFrame(data)


def _build_order_book(rows: int = 4) -> pd.DataFrame:
    price = 1.1
    entries: list[dict[str, float]] = []
    for level in range(rows):
        entries.append(
            {
                "bid_price": price - 0.0002 * (level + 1),
                "ask_price": price + 0.0002 * (level + 1),
                "bid_size": 1200 - level * 55,
                "ask_size": 1100 - level * 50,
            }
        )
    return pd.DataFrame(entries)


def test_how_sensor_emits_liquidity_audit() -> None:
    sensor = HowSensor()
    frame = _build_market_frame()

    signals = sensor.process(frame)

    assert len(signals) == 1
    signal = signals[0]
    assert signal.signal_type == "HOW"
    assert isinstance(signal.lineage, SensorLineageRecord)
    assert signal.lineage.dimension == "HOW"
    assert -1.0 <= float(signal.value["strength"]) <= 1.0
    metadata = signal.metadata or {}
    assert metadata.get("source") == "sensory.how"
    assert metadata.get("state") in {"nominal", "warning", "alert"}
    audit = metadata.get("audit")
    assert isinstance(audit, dict)
    assert set(audit.keys()) >= {"signal", "confidence", "liquidity", "participation"}
    lineage = metadata.get("lineage")
    assert isinstance(lineage, dict)
    assert lineage.get("dimension") == "HOW"
    assert lineage.get("source") == "sensory.how"
    assert lineage.get("inputs", {}).get("symbol") == "EURUSD"
    assert "liquidity" in lineage.get("telemetry", {})
    assert lineage.get("metadata", {}).get("mode") == "market_data"
    assert "state" in lineage.get("metadata", {})
    quality = metadata.get("quality")
    assert isinstance(quality, dict)
    assert quality.get("source") == "sensory.how"
    assert quality.get("confidence") == signal.confidence
    assert isinstance(quality.get("timestamp"), str)
    assert quality.get("state") == metadata.get("state")


def test_anomaly_sensor_sequence_mode_detects_spike() -> None:
    sensor = AnomalySensor()
    frame = _build_market_frame(anomaly_spike=True)

    signals = sensor.process(frame)
    assert len(signals) == 1
    signal = signals[0]
    assert signal.signal_type == "ANOMALY"
    assert isinstance(signal.lineage, SensorLineageRecord)
    assert signal.lineage.dimension == "ANOMALY"
    metadata = signal.metadata or {}
    assert metadata.get("source") == "sensory.anomaly"
    assert metadata.get("mode") == "sequence"
    assert metadata.get("thresholds") == {"warn": 0.4, "alert": 0.7}
    assert signal.value["strength"] >= 0.0
    assert signal.value["state"] in {"nominal", "warning", "alert"}
    assert signal.value["is_anomaly"] is True
    assert abs(float(signal.value["z_score"])) >= 0.0
    audit = metadata.get("audit")
    assert isinstance(audit, dict)
    assert audit.get("sample_size") >= sensor._config.sequence_min_length
    detector_meta = metadata.get("anomaly_detector")
    assert isinstance(detector_meta, dict)
    assert detector_meta.get("min_samples") == sensor._config.sequence_min_length
    assert detector_meta.get("sample_size") >= sensor._config.sequence_min_length
    lineage = metadata.get("lineage")
    assert isinstance(lineage, dict)
    assert lineage.get("metadata", {}).get("mode") == "sequence"
    assert lineage.get("inputs", {}).get("sequence_length") == 12
    assert "baseline" in lineage.get("telemetry", {})
    quality = metadata.get("quality")
    assert isinstance(quality, dict)
    assert quality.get("source") == "sensory.anomaly"
    assert quality.get("confidence") == signal.confidence
    assert isinstance(quality.get("timestamp"), str)
    assert quality.get("state") == metadata.get("state")


def test_anomaly_sensor_falls_back_to_market_payload() -> None:
    sensor = AnomalySensor()
    frame = _build_market_frame(rows=4)

    signals = sensor.process(frame)
    assert len(signals) == 1
    signal = signals[0]
    assert signal.signal_type == "ANOMALY"
    assert isinstance(signal.lineage, SensorLineageRecord)
    metadata = signal.metadata or {}
    assert metadata.get("mode") == "market_data"
    assert "baseline" in metadata.get("audit", {})
    lineage = metadata.get("lineage")
    assert isinstance(lineage, dict)
    assert lineage.get("metadata", {}).get("mode") == "market_data"
    assert lineage.get("inputs", {}).get("symbol") == "EURUSD"
    quality = metadata.get("quality")
    assert isinstance(quality, dict)
    assert quality.get("source") == "sensory.anomaly"
    assert quality.get("confidence") == signal.confidence
    assert isinstance(quality.get("timestamp"), str)
    assert quality.get("state") == metadata.get("state")


def test_how_sensor_threshold_state_escalates() -> None:
    engine = _StubHowEngine(strength=0.9, confidence=0.8, liquidity=0.6)
    sensor = HowSensor(engine=engine)
    frame = _build_market_frame()

    signal = sensor.process(frame)[0]

    assert signal.value["state"] == "alert"
    assert isinstance(signal.lineage, SensorLineageRecord)
    assert signal.lineage.dimension == "HOW"
    metadata = signal.metadata or {}
    assessment = metadata.get("threshold_assessment")
    assert assessment["breached_level"] == "alert"
    assert assessment["state"] == "alert"


def test_anomaly_sensor_threshold_state_escalates() -> None:
    engine = _StubAnomalyEngine(strength=0.65, confidence=0.9, baseline=0.2, latest=0.5)
    sensor = AnomalySensor(engine=engine)

    signal = sensor.process({"payload": "ignored"})[0]

    assert signal.value["state"] == "warning"
    assert signal.value["is_anomaly"] is False
    assert isinstance(signal.lineage, SensorLineageRecord)
    assert signal.lineage.dimension == "ANOMALY"
    metadata = signal.metadata or {}
    assessment = metadata.get("threshold_assessment")
    assert assessment["state"] == "warning"
    assert assessment["breached_level"] == "warn"


def test_how_sensor_records_lineage() -> None:
    recorder = SensorLineageRecorder(max_records=2)
    engine = _StubHowEngine(strength=0.55, confidence=0.7, liquidity=0.4, participation=0.6)
    sensor = HowSensor(engine=engine, lineage_recorder=recorder)
    frame = _build_market_frame()

    sensor.process(frame)
    sensor.process(frame)

    history = recorder.history()
    assert len(history) == 2
    latest = recorder.latest()
    assert latest is not None
    assert latest["dimension"] == "HOW"
    assert latest["source"] == "sensory.how"
    assert latest["outputs"]["signal"] == 0.55
    assert "liquidity" in latest["telemetry"]


def test_anomaly_sensor_records_lineage() -> None:
    recorder = SensorLineageRecorder(max_records=1)
    engine = _StubAnomalyEngine(strength=0.72, confidence=0.88, baseline=0.22, latest=0.51)
    sensor = AnomalySensor(engine=engine, lineage_recorder=recorder)

    signal = sensor.process({"key": "value"})[0]
    assert signal.signal_type == "ANOMALY"

    history = recorder.history()
    assert len(history) == 1
    entry = history[0]
    assert entry["dimension"] == "ANOMALY"
    assert entry["outputs"]["signal"] == 0.72
    assert entry["metadata"]["mode"] == "sequence"


def test_how_sensor_prefixes_order_book_metrics() -> None:
    engine = _StubHowEngine(
        strength=0.42,
        confidence=0.65,
        liquidity=0.51,
        participation=0.58,
        imbalance=0.12,
        volatility_drag=0.08,
    )
    sensor = HowSensor(engine=engine)
    frame = _build_market_frame()
    order_book = _build_order_book()

    signal = sensor.process(frame, order_book=order_book)[0]

    metadata = signal.metadata or {}
    audit = metadata.get("audit", {})
    assert audit["imbalance"] == 0.12
    assert "order_book_mid_price" in audit
    assert audit["order_book_mid_price"] > 0
    lineage = metadata.get("lineage")
    assert isinstance(lineage, dict)
    telemetry = lineage.get("telemetry", {})
    assert telemetry.get("imbalance") == 0.12
    assert "order_book_spread" in telemetry


def test_anomaly_sensor_sequence_filters_invalid_samples() -> None:
    engine = _RecordingAnomalyEngine()
    sensor = AnomalySensor(engine=engine)

    payload = [1.0, None, float("nan"), 2.0, "3.0"]

    signal = sensor.process(payload)[0]

    assert isinstance(engine.seen_payload, list)
    assert engine.seen_payload == [1.0, 2.0, 3.0]
    assert isinstance(signal.lineage, SensorLineageRecord)
    metadata = signal.metadata or {}
    assert metadata.get("dropped_samples") == 2
    lineage = metadata.get("lineage")
    assert isinstance(lineage, dict)
    assert lineage.get("metadata", {}).get("dropped_samples") == 2
    assert lineage.get("inputs", {}).get("sequence_length") == 3


def test_anomaly_sensor_clamps_confidence() -> None:
    engine = _RecordingAnomalyEngine()
    config = AnomalySensorConfig(minimum_confidence=0.35)
    sensor = AnomalySensor(config=config, engine=engine)

    signal = sensor.process([0.1] * 10)[0]

    assert signal.confidence == 0.35
    assert isinstance(signal.lineage, SensorLineageRecord)
    assert signal.metadata["threshold_assessment"]["state"] in {"nominal", "warning", "alert"}
