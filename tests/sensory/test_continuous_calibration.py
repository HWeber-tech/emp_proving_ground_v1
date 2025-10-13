from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

import pandas as pd

from src.sensory.calibration import ContinuousCalibrationConfig, ContinuousSensorCalibrator
from src.sensory.lineage import build_lineage_record
from src.sensory.how.how_sensor import HowSensor
from src.sensory.real_sensory_organ import RealSensoryOrgan
from src.sensory.signals import SensorSignal


def _quality_payload(signal: float, offset_seconds: float = 0.0) -> Mapping[str, Any]:
    timestamp = datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
    return {
        "timestamp": timestamp.isoformat(),
        "confidence": float(signal),
        "strength": float(signal),
    }


def _build_payload(signal: float) -> Mapping[str, Any]:
    return {
        "signal": float(signal),
        "confidence": 0.8,
        "value": {"strength": float(signal)},
        "quality": _quality_payload(signal),
    }


def _build_lineage(dimension: str, signal: float) -> Any:
    return build_lineage_record(
        dimension,
        f"test.{dimension.lower()}",
        inputs={"signal": float(signal)},
        outputs={"strength": float(signal), "confidence": 0.8},
    )


def test_calibrator_updates_how_sensor_thresholds() -> None:
    config = ContinuousCalibrationConfig(
        window=16,
        min_samples=5,
        recalibration_interval=1,
        warn_quantile=0.6,
        alert_quantile=0.9,
        lineage_staleness=timedelta(minutes=10),
    )
    calibrator = ContinuousSensorCalibrator(config)
    how_sensor = HowSensor()

    initial_warn = how_sensor.config.warn_threshold
    initial_alert = how_sensor.config.alert_threshold

    calibrator.register_dimension(
        "HOW",
        apply_callback=lambda update: how_sensor.recalibrate_thresholds(
            warn_threshold=update.warn_threshold,
            alert_threshold=update.alert_threshold,
        ),
    )

    samples = [0.15, 0.22, 0.35, 0.5, 0.62, 0.7, 0.78]
    for value in samples:
        calibrator.observe("HOW", _build_payload(value), _build_lineage("HOW", value))

    assert how_sensor.config.warn_threshold >= initial_warn
    assert how_sensor.config.alert_threshold >= initial_alert

    status = calibrator.status()
    how_state = status["dimensions"]["HOW"]
    assert how_state["last_update"] is not None
    assert how_state["last_update"]["warn_threshold"] is not None
    assert how_state["last_update"]["alert_threshold"] is not None


def test_calibrator_flags_lineage_issues() -> None:
    calibrator = ContinuousSensorCalibrator()
    calibrator.register_dimension("WHY", calibrate=False)

    calibrator.observe("WHY", _build_payload(0.2), lineage=None)

    status = calibrator.status()
    issues = status["dimensions"]["WHY"]["issues"]
    assert any("missing_lineage" in issue for issue in issues)
    assert any(entry.startswith("WHY:") for entry in status["issues"])


@dataclass
class _StubConfig:
    warn_threshold: float
    alert_threshold: float
    minimum_confidence: float = 0.2


class _StubSensor:
    def __init__(self, dimension: str, strength: float) -> None:
        self._dimension = dimension
        self._strength = strength

    def process(self, *_args: Any, **_kwargs: Any) -> list[SensorSignal]:
        lineage = _build_lineage(self._dimension, self._strength)
        metadata = {
            "source": f"stub.{self._dimension.lower()}",
            "quality": _quality_payload(self._strength),
            "lineage": lineage.as_dict(),
        }
        signal = SensorSignal(
            signal_type=self._dimension,
            value={"strength": self._strength},
            confidence=0.9,
            metadata=metadata,
            lineage=lineage,
        )
        return [signal]


class _CalibratedStubSensor(_StubSensor):
    def __init__(self, dimension: str, strength: float, warn: float, alert: float) -> None:
        super().__init__(dimension, strength)
        self._config = _StubConfig(warn_threshold=warn, alert_threshold=alert)

    @property
    def config(self) -> _StubConfig:
        return self._config

    def recalibrate_thresholds(
        self,
        *,
        warn_threshold: float | None = None,
        alert_threshold: float | None = None,
        minimum_confidence: float | None = None,
    ) -> None:
        if warn_threshold is not None:
            self._config.warn_threshold = float(warn_threshold)
        if alert_threshold is not None:
            self._config.alert_threshold = float(alert_threshold)
        if minimum_confidence is not None:
            self._config.minimum_confidence = float(minimum_confidence)


def test_real_sensory_organ_integrates_calibrator() -> None:
    calibrator = ContinuousSensorCalibrator(
        ContinuousCalibrationConfig(
            window=8,
            min_samples=3,
            recalibration_interval=1,
            warn_quantile=0.5,
            alert_quantile=0.9,
            lineage_staleness=timedelta(minutes=10),
        )
    )

    how_sensor = _CalibratedStubSensor("HOW", 0.65, warn=0.2, alert=0.5)
    anomaly_sensor = _CalibratedStubSensor("ANOMALY", 0.75, warn=0.3, alert=0.6)

    organ = RealSensoryOrgan(
        why_sensor=_StubSensor("WHY", 0.2),
        what_sensor=_StubSensor("WHAT", 0.3),
        when_sensor=_StubSensor("WHEN", 0.4),
        how_sensor=how_sensor,
        anomaly_sensor=anomaly_sensor,
        calibrator=calibrator,
    )

    frame = pd.DataFrame(
        {
            "timestamp": [datetime.now(timezone.utc)],
            "symbol": ["TEST"],
            "close": [1.0],
        }
    )

    for _ in range(3):
        organ.observe(frame)

    how_new_warn = how_sensor.config.warn_threshold
    anomaly_new_warn = anomaly_sensor.config.warn_threshold
    assert how_new_warn >= 0.2
    assert anomaly_new_warn >= 0.3

    status = organ.status()
    assert "calibration" in status
    how_state = status["calibration"]["dimensions"]["HOW"]
    assert how_state["last_update"] is not None
