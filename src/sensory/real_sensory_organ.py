from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import pandas as pd

from src.core.event_bus import Event
from src.sensory.anomaly.anomaly_sensor import AnomalySensor
from src.sensory.how.how_sensor import HowSensor
from src.sensory.lineage import build_lineage_record
from src.sensory.signals import IntegratedSignal, SensorSignal
from src.sensory.what.what_sensor import WhatSensor
from src.sensory.when.when_sensor import WhenSensor
from src.sensory.why.why_sensor import WhySensor

__all__ = ["RealSensoryOrgan"]


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
    ) -> None:
        self._why = why_sensor or WhySensor()
        self._how = how_sensor or HowSensor()
        self._what = what_sensor or WhatSensor()
        self._when = when_sensor or WhenSensor()
        self._anomaly = anomaly_sensor or AnomalySensor()
        self._event_bus = event_bus
        self._event_type = event_type
        self._audit_trail: deque[Mapping[str, Any]] = deque(maxlen=max(1, audit_window))
        self._latest_snapshot: Mapping[str, Any] | None = None

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

        audit_entry = {
            "symbol": resolved_symbol,
            "generated_at": timestamp.isoformat(),
            "unified_score": integrated.strength,
            "confidence": integrated.confidence,
            "dimensions": {
                name: self._summarise_dimension_payload(payload)
                for name, payload in dimension_payloads.items()
            },
        }
        self._audit_trail.appendleft(audit_entry)
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
                name: self._summarise_dimension_payload(entry)
                for name, entry in dimensions.items()
                if isinstance(entry, Mapping)
            }
        return payload

    def _summarise_dimension_payload(
        self, payload: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        summary: dict[str, Any] = {
            "signal": payload.get("signal"),
            "confidence": payload.get("confidence"),
        }

        metrics: dict[str, float] = {}
        value = payload.get("value")
        if isinstance(value, Mapping):
            metrics.update(self._extract_numeric_metrics(value))

        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            thresholds = metadata.get("thresholds")
            if isinstance(thresholds, Mapping):
                summary["thresholds"] = self._serialise_mapping(thresholds)

            audit_payload = metadata.get("audit")
            if isinstance(audit_payload, Mapping):
                metrics.update(self._extract_numeric_metrics(audit_payload))

        if metrics:
            summary["metrics"] = metrics

        return summary

    def _extract_numeric_metrics(self, mapping: Mapping[str, Any]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for key, value in mapping.items():
            if isinstance(value, Mapping):
                continue
            key_str = str(key)
            if key_str in {"strength", "confidence"}:
                continue
            number = self._as_float(value)
            if number is not None:
                metrics[key_str] = number
        return metrics

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

