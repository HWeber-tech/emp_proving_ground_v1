from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Any, Mapping, Sequence

import pandas as pd

from src.sensory.anomaly.basic_detector import AnomalyEvaluation, BasicAnomalyDetector
from src.sensory.enhanced.anomaly_dimension import AnomalyUnderstandingEngine
from src.sensory.lineage import (
    SensorLineageRecord,
    SensorLineageRecorder,
    build_lineage_record,
)
from src.sensory.signals import SensorSignal
from src.sensory.thresholds import ThresholdAssessment, evaluate_thresholds

__all__ = ["AnomalySensor", "AnomalySensorConfig"]


@dataclass(slots=True)
class AnomalySensorConfig:
    """Configuration for anomaly detection thresholds."""

    window: int = 32
    warn_threshold: float = 0.4
    alert_threshold: float = 0.7
    minimum_confidence: float = 0.2
    sequence_min_length: int = 8
    z_score_threshold: float = 3.0

    def clamp_confidence(self, confidence: float) -> float:
        return max(self.minimum_confidence, min(1.0, float(confidence)))


class AnomalySensor:
    """Detect displacements in price/volume behaviour for the ANOMALY dimension."""

    def __init__(
        self,
        config: AnomalySensorConfig | None = None,
        *,
        engine: AnomalyUnderstandingEngine | None = None,
        lineage_recorder: SensorLineageRecorder | None = None,
        basic_detector: BasicAnomalyDetector | None = None,
    ) -> None:
        self._engine = engine or AnomalyUnderstandingEngine()
        self._config = config or AnomalySensorConfig()
        self._lineage_recorder = lineage_recorder
        detector = basic_detector or BasicAnomalyDetector(
            window=self._config.window,
            min_samples=self._config.sequence_min_length,
            z_threshold=self._config.z_score_threshold,
        )
        self._basic_detector = detector

    @property
    def config(self) -> AnomalySensorConfig:
        """Expose the active anomaly sensor configuration."""

        return self._config

    def recalibrate_thresholds(
        self,
        *,
        warn_threshold: float | None = None,
        alert_threshold: float | None = None,
        minimum_confidence: float | None = None,
    ) -> None:
        """Apply recalibrated thresholds from the continuous calibrator."""

        warn_value = self._config.warn_threshold
        alert_value = self._config.alert_threshold

        if warn_threshold is not None:
            warn_value = max(0.0, min(1.0, float(warn_threshold)))

        if alert_threshold is not None:
            alert_value = max(0.0, min(1.0, float(alert_threshold)))

        if alert_value <= warn_value:
            alert_value = min(1.0, warn_value + 1e-6)

        self._config.warn_threshold = warn_value
        self._config.alert_threshold = alert_value

        if minimum_confidence is not None:
            confidence_value = max(0.0, min(1.0, float(minimum_confidence)))
            self._config.minimum_confidence = confidence_value

    def process(
        self, data: pd.DataFrame | Mapping[str, Any] | Sequence[float] | None
    ) -> list[SensorSignal]:
        if data is None:
            return [self._default_signal()]

        if isinstance(data, pd.DataFrame):
            return self._process_frame(data)

        inputs: Mapping[str, Any] | None = None
        if isinstance(data, Mapping):
            inputs = data
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            values, dropped = self._normalise_sequence(data)
            if not values:
                return [self._default_signal(reason="empty_sequence")]

            inputs = {"sequence_length": len(values), "latest": values[-1]}
            extra_metadata: dict[str, Any] | None = None
            if dropped:
                inputs["dropped_samples"] = dropped
                extra_metadata = {"dropped_samples": dropped}
            evaluation = self._basic_detector.evaluate(values)
            inputs["sample_size"] = evaluation.sample_size
            inputs["mean"] = evaluation.mean
            inputs["std_dev"] = evaluation.std_dev
            inputs["z_score"] = evaluation.z_score
            reading = self._engine.analyze_anomaly_understanding(values)
            return [
                self._build_signal(
                    reading,
                    mode_override="sequence",
                    inputs=inputs,
                    extra_metadata=extra_metadata,
                    basic_eval=evaluation,
                )
            ]
        else:
            inputs = {"payload_type": type(data).__name__}

        reading = self._engine.analyze_anomaly_understanding(data)
        return [self._build_signal(reading, mode_override=None, inputs=inputs)]

    # ------------------------------------------------------------------
    def _process_frame(self, df: pd.DataFrame) -> list[SensorSignal]:
        if df.empty or "close" not in df:
            return [self._default_signal(reason="empty_frame")]

        sequence = (
            df["close"].astype(float).tail(self._config.window).tolist()
            if "close" in df
            else []
        )
        evaluation: AnomalyEvaluation | None = None
        if sequence:
            evaluation = self._basic_detector.evaluate(sequence)
        if len(sequence) >= self._config.sequence_min_length:
            reading = self._engine.analyze_anomaly_understanding(sequence)
            inputs = {"sequence_length": len(sequence), "latest": sequence[-1]}
            if evaluation is not None:
                inputs["sample_size"] = evaluation.sample_size
                inputs["mean"] = evaluation.mean
                inputs["std_dev"] = evaluation.std_dev
                inputs["z_score"] = evaluation.z_score
            return [
                self._build_signal(
                    reading,
                    mode_override="sequence",
                    inputs=inputs,
                    basic_eval=evaluation,
                )
            ]

        payload = self._build_market_payload(df)
        reading = self._engine.analyze_anomaly_understanding(payload)
        return [
            self._build_signal(
                reading,
                mode_override="market_data",
                inputs=payload,
                basic_eval=evaluation,
            )
        ]

    def _build_market_payload(self, df: pd.DataFrame) -> Mapping[str, Any]:
        row = df.iloc[-1]
        payload = {
            "timestamp": row.get("timestamp"),
            "symbol": row.get("symbol", "UNKNOWN"),
            "open": float(row.get("open", row.get("close", 0.0) or 0.0)),
            "high": float(row.get("high", row.get("close", 0.0) or 0.0)),
            "low": float(row.get("low", row.get("close", 0.0) or 0.0)),
            "close": float(row.get("close", 0.0) or 0.0),
            "volume": float(row.get("volume", 0.0) or 0.0),
            "volatility": float(row.get("volatility", 0.0) or 0.0),
            "spread": float(row.get("spread", 0.0) or 0.0),
            "data_quality": float(row.get("data_quality", 0.8) or 0.8),
        }
        return payload

    def _build_signal(
        self,
        reading_adapter,
        *,
        mode_override: str | None,
        inputs: Mapping[str, Any] | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
        basic_eval: AnomalyEvaluation | None = None,
    ) -> SensorSignal:
        reading = reading_adapter.reading
        signal_strength = float(getattr(reading, "signal_strength", 0.0))
        confidence = self._config.clamp_confidence(getattr(reading, "confidence", 0.0))
        context = dict(getattr(reading, "context", {}) or {})
        mode = mode_override or reading_adapter.get("mode", "sequence")

        assessment = evaluate_thresholds(
            signal_strength,
            self._config.warn_threshold,
            self._config.alert_threshold,
            mode="positive",
        )

        baseline = float(reading_adapter.get("baseline", 0.0))
        dispersion = float(reading_adapter.get("dispersion", 0.0))
        latest = float(reading_adapter.get("latest", 0.0))
        if basic_eval is not None:
            baseline = basic_eval.mean
            dispersion = basic_eval.std_dev
            latest = basic_eval.latest

        if dispersion <= 0.0:
            z_score = basic_eval.z_score if basic_eval is not None else 0.0
        else:
            z_score = (latest - baseline) / dispersion

        if basic_eval is not None and dispersion <= 0.0:
            z_score = basic_eval.z_score
        abs_z_score = abs(z_score)
        telemetry: dict[str, float] = {
            "baseline": baseline,
            "dispersion": dispersion,
            "latest": latest,
            "z_score": float(z_score),
        }
        if basic_eval is not None:
            telemetry["sample_size"] = float(basic_eval.sample_size)

        anomaly_flag = basic_eval.is_anomaly if basic_eval is not None else abs_z_score >= self._config.z_score_threshold
        if assessment.state in {"alert", "critical"}:
            anomaly_flag = True

        lineage = build_lineage_record(
            "ANOMALY",
            "sensory.anomaly",
            inputs=inputs or {},
            outputs={"signal": signal_strength, "confidence": confidence},
            telemetry=telemetry,
            metadata={
                "mode": mode,
                "thresholds": assessment.thresholds,
                "state": assessment.state,
                "breached_level": assessment.breached_level,
                **(extra_metadata or {}),
            },
        )

        metadata: dict[str, object] = {
            "source": "sensory.anomaly",
            "mode": mode,
            "thresholds": {
                "warn": self._config.warn_threshold,
                "alert": self._config.alert_threshold,
            },
            "audit": {
                "signal": signal_strength,
                "confidence": confidence,
                "context": context,
                "z_score": float(z_score),
            },
            "lineage": lineage.as_dict(),
            "state": assessment.state,
            "threshold_assessment": assessment.as_dict(),
            "is_anomaly": anomaly_flag,
        }
        if basic_eval is not None:
            metadata["audit"]["sample_size"] = basic_eval.sample_size
            metadata["audit"]["mean"] = basic_eval.mean
            metadata["audit"]["std_dev"] = basic_eval.std_dev
            metadata["audit"]["latest"] = basic_eval.latest
            metadata["anomaly_detector"] = {
                "window": self._basic_detector.window,
                "min_samples": self._basic_detector.min_samples,
                "z_threshold": self._basic_detector.z_threshold,
                "sample_size": basic_eval.sample_size,
            }
        metadata["audit"].update(telemetry)
        if extra_metadata:
            metadata.update(extra_metadata)

        timestamp = self._resolve_timestamp(inputs)
        quality: dict[str, object] = {
            "source": "sensory.anomaly",
            "timestamp": timestamp.isoformat(),
            "confidence": confidence,
            "state": assessment.state,
        }
        data_quality = self._extract_data_quality(inputs)
        if data_quality is not None:
            quality["data_quality"] = data_quality
        metadata["quality"] = quality

        value: dict[str, object] = {
            "strength": signal_strength,
            "confidence": confidence,
            "context": context,
            "state": assessment.state,
            "is_anomaly": anomaly_flag,
            "z_score": float(z_score),
        }
        if basic_eval is not None:
            value["sample_size"] = basic_eval.sample_size
        self._record_lineage(lineage)
        return SensorSignal(
            signal_type="ANOMALY",
            value=value,
            confidence=confidence,
            metadata=metadata,
            lineage=lineage,
        )

    def _default_signal(self, *, reason: str | None = None) -> SensorSignal:
        thresholds = {
            "warn": self._config.warn_threshold,
            "alert": self._config.alert_threshold,
        }
        assessment = ThresholdAssessment(
            state="nominal",
            magnitude=0.0,
            thresholds=thresholds,
            breached_level=None,
            breach_ratio=0.0,
            distance_to_warn=thresholds["warn"],
            distance_to_alert=thresholds["alert"],
        )
        lineage = build_lineage_record(
            "ANOMALY",
            "sensory.anomaly",
            inputs={},
            outputs={"signal": 0.0, "confidence": 0.0},
            telemetry={},
            metadata={
                "mode": "default",
                "thresholds": thresholds,
                "state": assessment.state,
            },
        )

        metadata: dict[str, object] = {
            "source": "sensory.anomaly",
            "thresholds": thresholds,
            "mode": "unknown",
            "audit": {
                "signal": 0.0,
                "confidence": 0.0,
                "z_score": 0.0,
                "baseline": 0.0,
                "dispersion": 0.0,
                "latest": 0.0,
            },
            "lineage": lineage.as_dict(),
            "state": assessment.state,
            "threshold_assessment": assessment.as_dict(),
            "is_anomaly": False,
        }
        if reason is not None:
            metadata["failure_reason"] = reason
        metadata["quality"] = {
            "source": "sensory.anomaly",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence": 0.0,
            "state": assessment.state,
        }
        self._record_lineage(lineage)
        return SensorSignal(
            signal_type="ANOMALY",
            value={
                "strength": 0.0,
                "confidence": 0.0,
                "state": assessment.state,
                "is_anomaly": False,
                "z_score": 0.0,
            },
            confidence=0.0,
            metadata=metadata,
            lineage=lineage,
        )

    def _record_lineage(self, lineage: SensorLineageRecord) -> None:
        if self._lineage_recorder is not None:
            self._lineage_recorder.record(lineage)

    def _normalise_sequence(
        self, values: Sequence[Any]
    ) -> tuple[list[float], int]:
        cleaned: list[float] = []
        dropped = 0
        for item in values:
            try:
                value = float(item)
            except (TypeError, ValueError):
                dropped += 1
                continue
            if not isfinite(value):
                dropped += 1
                continue
            cleaned.append(value)
        return cleaned, dropped

    def _resolve_timestamp(self, inputs: Mapping[str, Any] | None) -> datetime:
        if not inputs:
            return datetime.now(timezone.utc)

        candidate = inputs.get("timestamp")
        if isinstance(candidate, datetime):
            if candidate.tzinfo is None:
                return candidate.replace(tzinfo=timezone.utc)
            return candidate.astimezone(timezone.utc)

        try:
            timestamp = pd.to_datetime(candidate, utc=True, errors="coerce")
        except Exception:
            timestamp = None

        if timestamp is None or timestamp is pd.NaT:
            return datetime.now(timezone.utc)
        return timestamp.to_pydatetime()

    def _extract_data_quality(self, inputs: Mapping[str, Any] | None) -> float | None:
        if not isinstance(inputs, Mapping):
            return None
        candidate = inputs.get("data_quality")
        try:
            if candidate is not None:
                return float(candidate)
        except (TypeError, ValueError):
            return None
        return None
