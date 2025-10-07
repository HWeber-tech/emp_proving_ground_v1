from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import pandas as pd

from src.core.event_bus import Event
from src.sensory.anomaly.anomaly_sensor import AnomalySensor
from src.sensory.how.how_sensor import HowSensor
from src.sensory.lineage import build_lineage_record
from src.sensory.lineage_publisher import SensoryLineagePublisher
from src.sensory.monitoring.sensor_drift import SensorDriftSummary, evaluate_sensor_drift
from src.sensory.signals import IntegratedSignal, SensorSignal
from src.sensory.what.what_sensor import WhatSensor
from src.sensory.when.when_sensor import WhenSensor
from src.sensory.why.why_sensor import WhySensor

__all__ = ["RealSensoryOrgan", "SensoryDriftConfig"]


@dataclass(slots=True)
class SensoryDriftConfig:
    """Configuration controlling sensory drift instrumentation."""

    baseline_window: int = 60
    evaluation_window: int = 20
    min_observations: int = 10
    z_threshold: float = 3.0
    sensors: tuple[str, ...] = ("WHY", "WHAT", "WHEN", "HOW", "ANOMALY")

    def required_samples(self) -> int:
        return self.baseline_window + self.evaluation_window


class RealSensoryOrgan:
    """Integrate canonical sensory organs into a single executable surface."""

    def __init__(
        self,
        *,
        why_sensor: WhySensor | None = None,
        how_sensor: HowSensor | None = None,
        what_sensor: WhatSensor | None = None,
        when_sensor: WhenSensor | None = None,
        anomaly_sensor: AnomalySensor | None = None,
        event_bus: Any | None = None,
        event_type: str = "telemetry.sensory.snapshot",
        audit_window: int = 128,
        drift_config: SensoryDriftConfig | None = None,
        lineage_publisher: SensoryLineagePublisher | None = None,
    ) -> None:
        self._why = why_sensor or WhySensor()
        self._how = how_sensor or HowSensor()
        self._what = what_sensor or WhatSensor()
        self._when = when_sensor or WhenSensor()
        self._anomaly = anomaly_sensor or AnomalySensor()
        self._event_bus = event_bus
        self._event_type = event_type
        self._drift_config = self._validate_drift_config(drift_config)
        min_window = max(self._validate_window(audit_window), self._drift_config.required_samples())
        self._audit_trail: deque[Mapping[str, Any]] = deque(maxlen=min_window)
        self._latest_snapshot: Mapping[str, Any] | None = None
        self._latest_drift: SensorDriftSummary | None = None
        self._lineage_publisher = lineage_publisher

    # ------------------------------------------------------------------
    def observe(
        self,
        market_data: pd.DataFrame | None,
        *,
        symbol: str | None = None,
        order_book: pd.DataFrame | None = None,
        option_positions: pd.DataFrame | None = None,
        narrative_events: Sequence[Any] | None = None,
        macro_regime_flags: Mapping[str, float] | None = None,
        macro_events: Sequence[datetime] | None = None,
        anomaly_input: pd.DataFrame | Mapping[str, Any] | Sequence[float] | None = None,
        as_of: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Produce a fused sensory snapshot from canonical organs."""

        frame = market_data if isinstance(market_data, pd.DataFrame) else pd.DataFrame()
        resolved_symbol = self._resolve_symbol(symbol, frame)
        timestamp = self._resolve_timestamp(as_of, frame)

        why_signal = self._first_signal(
            self._why.process(
                frame,
                narrative_events=list(narrative_events or []),
                macro_regime_flags=macro_regime_flags or {},
                as_of=timestamp,
            ),
            "WHY",
        )
        what_signal = self._first_signal(self._what.process(frame), "WHAT")
        when_signal = self._first_signal(
            self._when.process(
                frame,
                option_positions=option_positions,
                macro_events=list(macro_events or []),
            ),
            "WHEN",
        )
        how_signal = self._first_signal(
            self._how.process(
                frame,
                order_book=order_book,
            ),
            "HOW",
        )
        anomaly_payload = anomaly_input if anomaly_input is not None else frame
        anomaly_signal = self._first_signal(
            self._anomaly.process(anomaly_payload),
            "ANOMALY",
        )

        signals = {
            "WHY": why_signal,
            "WHAT": what_signal,
            "WHEN": when_signal,
            "HOW": how_signal,
            "ANOMALY": anomaly_signal,
        }

        dimension_payloads: dict[str, dict[str, Any]] = {}
        for name, signal in signals.items():
            dimension_payloads[name] = {
                "signal": self._extract_strength(signal),
                "confidence": float(signal.confidence),
                "value": self._serialise_value(signal.value),
                "metadata": self._serialise_mapping(signal.metadata),
            }

        integrated = self._build_integrated_signal(signals)
        lineage = build_lineage_record(
            "SENSORY_FUSION",
            "sensory.real_organ",
            inputs={
                name: {
                    "signal": payload["signal"],
                    "confidence": payload["confidence"],
                }
                for name, payload in dimension_payloads.items()
            },
            outputs={
                "strength": integrated.strength,
                "confidence": integrated.confidence,
            },
            telemetry={"contributing": integrated.contributing},
            metadata={"symbol": resolved_symbol},
        )

        snapshot = {
            "generated_at": timestamp,
            "symbol": resolved_symbol,
            "integrated_signal": integrated,
            "dimensions": dimension_payloads,
            "metadata": dict(metadata or {}),
            "lineage": lineage,
        }

        self._record_sensor_lineage(
            dimension_payloads,
            symbol=resolved_symbol,
            generated_at=timestamp,
        )

        audit_entry = {
            "symbol": resolved_symbol,
            "generated_at": timestamp.isoformat(),
            "unified_score": integrated.strength,
            "confidence": integrated.confidence,
            "dimensions": {
                name: {
                    "signal": payload["signal"],
                    "confidence": payload["confidence"],
                }
                for name, payload in dimension_payloads.items()
            },
        }
        self._audit_trail.appendleft(audit_entry)
        self._refresh_drift_summary()
        snapshot["drift_summary"] = self._latest_drift
        self._latest_snapshot = snapshot
        self._publish_snapshot(snapshot)
        return snapshot

    # ------------------------------------------------------------------
    def audit_trail(self, limit: int = 20) -> list[Mapping[str, Any]]:
        """Return recent sensory audit entries (most recent first)."""

        capped = max(0, limit)
        return [dict(entry) for entry in list(self._audit_trail)[:capped]]

    def status(self) -> Mapping[str, Any]:
        """Expose latest snapshot metadata for runtime summaries."""

        latest = self._latest_snapshot
        return {
            "samples": len(self._audit_trail),
            "latest": self._serialise_snapshot(latest) if latest else None,
            "sensor_audit": self.audit_trail(limit=5),
            "drift_summary": self._serialise_drift_summary(self._latest_drift),
        }

    def metrics(self) -> Mapping[str, Any]:
        """Expose dimension-level metrics for runtime summaries and telemetry."""

        latest_snapshot = self._latest_snapshot
        serialised_latest = self._serialise_snapshot(latest_snapshot)

        dimensions: dict[str, Mapping[str, Any]] = {}
        if isinstance(latest_snapshot, Mapping):
            raw_dimensions = latest_snapshot.get("dimensions")
            if isinstance(raw_dimensions, Mapping):
                for name, payload in raw_dimensions.items():
                    if not isinstance(name, str) or not isinstance(payload, Mapping):
                        continue
                    dimensions[name] = self._build_dimension_metrics(payload)

        integrated_metrics: Mapping[str, Any] | None = None
        if isinstance(latest_snapshot, Mapping):
            integrated = latest_snapshot.get("integrated_signal")
            if isinstance(integrated, IntegratedSignal):
                integrated_metrics = {
                    "strength": integrated.strength,
                    "confidence": integrated.confidence,
                    "direction": integrated.direction,
                }

        return {
            "symbol": (serialised_latest or {}).get("symbol"),
            "generated_at": (serialised_latest or {}).get("generated_at"),
            "samples": len(self._audit_trail),
            "integrated": integrated_metrics,
            "dimensions": dimensions,
            "drift_summary": self._serialise_drift_summary(self._latest_drift),
        }

    # ------------------------------------------------------------------
    def _first_signal(self, signals: Sequence[SensorSignal], dimension: str) -> SensorSignal:
        for signal in signals:
            if isinstance(signal, SensorSignal):
                return signal
        return SensorSignal(signal_type=dimension, value={"strength": 0.0}, confidence=0.0)

    def _resolve_symbol(self, symbol: str | None, frame: pd.DataFrame) -> str:
        if symbol:
            return str(symbol)
        if not frame.empty and "symbol" in frame:
            value = frame["symbol"].iloc[-1]
            if isinstance(value, str):
                return value
        return "UNKNOWN"

    def _resolve_timestamp(self, as_of: datetime | None, frame: pd.DataFrame) -> datetime:
        if isinstance(as_of, datetime):
            ts = as_of
        elif not frame.empty and "timestamp" in frame:
            ts_candidate = pd.to_datetime(frame["timestamp"].iloc[-1], utc=True, errors="coerce")
            if ts_candidate is not None and not pd.isna(ts_candidate):
                ts = ts_candidate.to_pydatetime()
            else:
                ts = datetime.now(timezone.utc)
        else:
            ts = datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts

    def _extract_strength(self, signal: SensorSignal) -> float:
        value = signal.value
        candidate: float | None = None
        if isinstance(value, Mapping):
            if "strength" in value:
                candidate = self._as_float(value.get("strength"))
            elif signal.signal_type == "WHAT":
                candidate = self._as_float(value.get("pattern_strength"))
        if candidate is None:
            candidate = self._as_float(getattr(signal, "strength", None))
        return candidate if candidate is not None else 0.0

    def _build_integrated_signal(self, signals: Mapping[str, SensorSignal]) -> IntegratedSignal:
        strengths: list[tuple[float, float, str]] = []
        for name, signal in signals.items():
            strength = self._extract_strength(signal)
            confidence = max(0.0, min(1.0, float(signal.confidence)))
            strengths.append((strength, confidence, name))

        if not strengths:
            return IntegratedSignal(direction=0.0, strength=0.0, confidence=0.0, contributing=[])

        weighted_total = sum(strength * confidence for strength, confidence, _ in strengths)
        total_confidence = sum(confidence for _, confidence, _ in strengths)
        if total_confidence > 0:
            fused_strength = weighted_total / total_confidence
            fused_confidence = total_confidence / len(strengths)
        else:
            fused_strength = sum(strength for strength, _, _ in strengths) / len(strengths)
            fused_confidence = 0.0

        direction = 0.0
        if fused_strength > 0:
            direction = 1.0
        elif fused_strength < 0:
            direction = -1.0

        contributing = [name for _, _, name in strengths]
        return IntegratedSignal(
            direction=direction,
            strength=float(fused_strength),
            confidence=float(max(0.0, min(1.0, fused_confidence))),
            contributing=contributing,
        )

    def _serialise_value(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): self._serialise_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._serialise_value(item) for item in value]
        return value

    def _serialise_mapping(self, mapping: Mapping[str, Any] | None) -> Mapping[str, Any]:
        if not mapping:
            return {}
        return {str(key): self._serialise_value(val) for key, val in mapping.items()}

    def _record_sensor_lineage(
        self,
        dimension_payloads: Mapping[str, Mapping[str, Any]],
        *,
        symbol: str,
        generated_at: datetime,
    ) -> None:
        publisher = self._lineage_publisher
        if publisher is None:
            return

        for dimension, payload in dimension_payloads.items():
            raw_metadata = payload.get("metadata")
            metadata = raw_metadata if isinstance(raw_metadata, Mapping) else None
            if not metadata:
                continue
            lineage_payload = metadata.get("lineage")
            if lineage_payload is None:
                continue

            state_value = metadata.get("state")
            state = state_value if isinstance(state_value, str) else None

            threshold_state: str | None = None
            threshold_metadata = metadata.get("threshold_assessment")
            if isinstance(threshold_metadata, Mapping):
                threshold_state_value = threshold_metadata.get("state")
                if isinstance(threshold_state_value, str):
                    threshold_state = threshold_state_value

            extra_metadata = {k: v for k, v in metadata.items() if k != "lineage"}

            publisher.record(
                dimension,
                lineage_payload,
                symbol=symbol,
                generated_at=generated_at,
                strength=payload.get("signal"),
                confidence=payload.get("confidence"),
                state=state,
                threshold_state=threshold_state,
                metadata=extra_metadata,
            )

    def _serialise_snapshot(self, snapshot: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if snapshot is None:
            return None
        payload: dict[str, Any] = {
            "symbol": snapshot.get("symbol"),
            "generated_at": self._format_timestamp(snapshot.get("generated_at")),
        }
        integrated = snapshot.get("integrated_signal")
        if isinstance(integrated, IntegratedSignal):
            payload["integrated_signal"] = {
                "direction": integrated.direction,
                "strength": integrated.strength,
                "confidence": integrated.confidence,
                "contributing": list(integrated.contributing),
            }
        dimensions = snapshot.get("dimensions")
        if isinstance(dimensions, Mapping):
            payload["dimensions"] = {
                name: {
                    "signal": entry.get("signal"),
                    "confidence": entry.get("confidence"),
                }
                for name, entry in dimensions.items()
                if isinstance(entry, Mapping)
            }
        drift_summary = snapshot.get("drift_summary")
        if isinstance(drift_summary, SensorDriftSummary):
            payload["drift_summary"] = self._serialise_drift_summary(drift_summary)
        return payload

    def _build_dimension_metrics(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        signal = self._as_float(payload.get("signal"))
        confidence = self._as_float(payload.get("confidence"))
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
        state = metadata.get("state") if isinstance(metadata.get("state"), str) else None
        threshold_state: str | None = None
        thresholds = metadata.get("threshold_assessment")
        if isinstance(thresholds, Mapping):
            value = thresholds.get("state")
            if isinstance(value, str):
                threshold_state = value
        return {
            "signal": signal,
            "confidence": confidence,
            "state": state,
            "threshold_state": threshold_state,
        }

    def _format_timestamp(self, value: Any) -> str | None:
        if isinstance(value, datetime):
            return value.isoformat()
        try:
            converted = pd.to_datetime(value, utc=True, errors="coerce")
        except Exception:
            return None
        if converted is pd.NaT:
            return None
        return converted.isoformat()

    def _as_float(self, value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _validate_window(self, value: int) -> int:
        if value <= 0:
            return 1
        return int(value)

    def _validate_drift_config(
        self, drift_config: SensoryDriftConfig | None
    ) -> SensoryDriftConfig:
        config = drift_config or SensoryDriftConfig()
        if config.baseline_window <= 0 or config.evaluation_window <= 0:
            raise ValueError("Drift windows must be positive integers")
        if config.min_observations <= 0:
            raise ValueError("min_observations must be a positive integer")
        if not config.sensors:
            raise ValueError("At least one sensor must be configured for drift monitoring")
        return config

    def _build_drift_frame(self, config: SensoryDriftConfig) -> pd.DataFrame | None:
        sensors = list(dict.fromkeys(config.sensors))
        if not sensors:
            return None

        records: list[dict[str, Any]] = []
        for entry in reversed(self._audit_trail):
            dimensions = entry.get("dimensions", {})
            if not isinstance(dimensions, Mapping):
                continue
            record: dict[str, Any] = {"generated_at": entry.get("generated_at")}
            for sensor in sensors:
                payload = dimensions.get(sensor)
                if isinstance(payload, Mapping):
                    record[sensor] = payload.get("signal")
            records.append(record)

        if not records:
            return None

        frame = pd.DataFrame(records)
        sensor_columns = [sensor for sensor in sensors if sensor in frame.columns]
        if not sensor_columns:
            return None

        for column in sensor_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame = frame.dropna(subset=sensor_columns, how="all")
        required = config.required_samples()
        if frame.shape[0] < required:
            return None
        return frame.tail(required)

    def _refresh_drift_summary(self) -> None:
        frame = self._build_drift_frame(self._drift_config)
        if frame is None:
            self._latest_drift = None
            return

        try:
            summary = evaluate_sensor_drift(
                frame,
                sensor_columns=list(self._drift_config.sensors),
                baseline_window=self._drift_config.baseline_window,
                evaluation_window=self._drift_config.evaluation_window,
                min_observations=self._drift_config.min_observations,
                z_threshold=self._drift_config.z_threshold,
            )
        except ValueError:
            self._latest_drift = None
            return

        self._latest_drift = summary

    def _serialise_drift_summary(
        self, summary: SensorDriftSummary | None
    ) -> Mapping[str, Any] | None:
        if summary is None:
            return None
        data = summary.as_dict()
        data["exceeded"] = [result.as_dict() for result in summary.exceeded]
        return data

    def _publish_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        if self._event_bus is None:
            return

        event_payload = {
            "symbol": snapshot.get("symbol"),
            "generated_at": self._format_timestamp(snapshot.get("generated_at")),
            "integrated_signal": {
                "direction": snapshot["integrated_signal"].direction
                if isinstance(snapshot.get("integrated_signal"), IntegratedSignal)
                else None,
                "strength": snapshot["integrated_signal"].strength
                if isinstance(snapshot.get("integrated_signal"), IntegratedSignal)
                else None,
                "confidence": snapshot["integrated_signal"].confidence
                if isinstance(snapshot.get("integrated_signal"), IntegratedSignal)
                else None,
                "contributing": snapshot["integrated_signal"].contributing
                if isinstance(snapshot.get("integrated_signal"), IntegratedSignal)
                else [],
            },
            "dimensions": snapshot.get("dimensions"),
            "metadata": snapshot.get("metadata"),
            "lineage": snapshot.get("lineage").as_dict()
            if hasattr(snapshot.get("lineage"), "as_dict")
            else snapshot.get("lineage"),
            "drift_summary": self._serialise_drift_summary(
                snapshot.get("drift_summary")
                if isinstance(snapshot.get("drift_summary"), SensorDriftSummary)
                else None
            ),
        }

        event = Event(
            type=self._event_type,
            payload=event_payload,
            source="sensory.real_organ",
        )

        publish_from_sync = getattr(self._event_bus, "publish_from_sync", None)
        if callable(publish_from_sync):
            try:
                publish_from_sync(event)
                return
            except Exception:
                return
        publish_sync = getattr(self._event_bus, "publish_sync", None)
        if callable(publish_sync):
            try:
                publish_sync(self._event_type, event_payload, source="sensory.real_organ")
            except Exception:
                return
