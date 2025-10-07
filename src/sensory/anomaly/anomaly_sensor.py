from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

import pandas as pd

from src.sensory.enhanced.anomaly_dimension import AnomalyIntelligenceEngine
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

    def clamp_confidence(self, confidence: float) -> float:
        return max(self.minimum_confidence, min(1.0, float(confidence)))


class AnomalySensor:
    """Detect displacements in price/volume behaviour for the ANOMALY dimension."""

    def __init__(
        self,
        config: AnomalySensorConfig | None = None,
        *,
        engine: AnomalyIntelligenceEngine | None = None,
        lineage_recorder: SensorLineageRecorder | None = None,
    ) -> None:
        self._engine = engine or AnomalyIntelligenceEngine()
        self._config = config or AnomalySensorConfig()
        self._lineage_recorder = lineage_recorder

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
            reading = self._engine.analyze_anomaly_intelligence(values)
            return [
                self._build_signal(
                    reading,
                    mode_override="sequence",
                    inputs=inputs,
                    extra_metadata=extra_metadata,
                )
            ]
        else:
            inputs = {"payload_type": type(data).__name__}

        reading = self._engine.analyze_anomaly_intelligence(data)
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
        if len(sequence) >= self._config.sequence_min_length:
            reading = self._engine.analyze_anomaly_intelligence(sequence)
            inputs = {"sequence_length": len(sequence), "latest": sequence[-1]}
            return [
                self._build_signal(
                    reading,
                    mode_override="sequence",
                    inputs=inputs,
                )
            ]

        payload = self._build_market_payload(df)
        reading = self._engine.analyze_anomaly_intelligence(payload)
        return [
            self._build_signal(
                reading,
                mode_override="market_data",
                inputs=payload,
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
        }
        return payload

    def _build_signal(
        self,
        reading_adapter,
        *,
        mode_override: str | None,
        inputs: Mapping[str, Any] | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
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

        telemetry: dict[str, float] = {
            "baseline": float(reading_adapter.get("baseline", 0.0)),
            "dispersion": float(reading_adapter.get("dispersion", 0.0)),
            "latest": float(reading_adapter.get("latest", 0.0)),
        }

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
            },
            "lineage": lineage.as_dict(),
            "state": assessment.state,
            "threshold_assessment": assessment.as_dict(),
        }
        metadata["audit"].update(telemetry)
        if extra_metadata:
            metadata.update(extra_metadata)

        value: dict[str, object] = {
            "strength": signal_strength,
            "confidence": confidence,
            "context": context,
            "state": assessment.state,
        }
        self._record_lineage(lineage)
        return SensorSignal(
            signal_type="ANOMALY",
            value=value,
            confidence=confidence,
            metadata=metadata,
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
            "audit": {"signal": 0.0, "confidence": 0.0},
            "lineage": lineage.as_dict(),
            "state": assessment.state,
            "threshold_assessment": assessment.as_dict(),
        }
        if reason is not None:
            metadata["failure_reason"] = reason
        self._record_lineage(lineage)
        return SensorSignal(
            signal_type="ANOMALY",
            value={
                "strength": 0.0,
                "confidence": 0.0,
                "state": assessment.state,
            },
            confidence=0.0,
            metadata=metadata,
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
