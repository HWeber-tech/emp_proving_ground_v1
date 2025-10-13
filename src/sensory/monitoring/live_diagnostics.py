"""Live sensory diagnostics built from market data frames or data managers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, MutableMapping, Sequence, TYPE_CHECKING

import pandas as pd

from src.sensory.monitoring.sensor_drift import SensorDriftSummary
from src.sensory.real_sensory_organ import RealSensoryOrgan
from src.sensory.signals import IntegratedSignal
from src.sensory.why.narrative_hooks import NarrativeEvent

if TYPE_CHECKING:  # pragma: no cover - optional institutional dependency
    from src.data_integration.real_data_integration import RealDataManager

__all__ = [
    "LiveSensoryDiagnostics",
    "build_live_sensory_diagnostics",
    "build_live_sensory_diagnostics_from_manager",
]


@dataclass(frozen=True)
class LiveSensoryDiagnostics:
    """Structured diagnostics summarising live sensory posture."""

    symbol: str
    generated_at: datetime
    samples: int
    integrated_signal: Mapping[str, Any]
    anomaly: Mapping[str, Any]
    drift_summary: Mapping[str, Any] | None
    why_quality: Mapping[str, Any]
    why_explanation: str
    dimensions: Mapping[str, Mapping[str, Any]]

    def as_dict(self) -> Mapping[str, Any]:
        """Serialise the diagnostics into JSON-friendly primitives."""

        return {
            "symbol": self.symbol,
            "generated_at": self.generated_at.isoformat(),
            "samples": self.samples,
            "integrated_signal": dict(self.integrated_signal),
            "anomaly": dict(self.anomaly),
            "drift_summary": dict(self.drift_summary) if self.drift_summary else None,
            "why_quality": dict(self.why_quality),
            "why_explanation": self.why_explanation,
            "dimensions": {name: dict(payload) for name, payload in self.dimensions.items()},
        }


def build_live_sensory_diagnostics(
    market_data: pd.DataFrame,
    *,
    symbol: str | None = None,
    organ: RealSensoryOrgan | None = None,
    narrative_events: Sequence[NarrativeEvent] | None = None,
    macro_regime_flags: Mapping[str, float] | None = None,
) -> LiveSensoryDiagnostics:
    """Replay market data into the sensory cortex and derive diagnostics."""

    if market_data is None or market_data.empty:
        raise ValueError("market_data must contain rows for sensory diagnostics")

    resolved_symbol = _resolve_symbol(symbol, market_data)
    sensory_organ = organ or RealSensoryOrgan()
    events = list(narrative_events or [])

    frame = market_data.copy()
    if "timestamp" in frame.columns:
        frame = frame.sort_values("timestamp")

    snapshot: Mapping[str, Any] | None = None
    for idx in range(frame.shape[0]):
        window = frame.iloc[: idx + 1]
        snapshot = sensory_organ.observe(
            window,
            symbol=resolved_symbol,
            narrative_events=events,
            macro_regime_flags=macro_regime_flags or {},
        )

    if snapshot is None:
        snapshot = sensory_organ.observe(
            frame,
            symbol=resolved_symbol,
            narrative_events=events,
            macro_regime_flags=macro_regime_flags or {},
        )

    metrics = sensory_organ.metrics()
    samples = int(metrics.get("samples", frame.shape[0]))

    generated_at = _ensure_datetime(snapshot.get("generated_at"))
    integrated_signal = _serialise_integrated_signal(snapshot.get("integrated_signal"))

    raw_dimensions = snapshot.get("dimensions")
    dimensions = _extract_dimensions(raw_dimensions)

    anomaly_payload = dimensions.get("ANOMALY", {})
    drift_summary = _serialise_drift_summary(snapshot.get("drift_summary"))

    why_payload = dimensions.get("WHY", {})
    why_quality = _normalise_mapping(why_payload.get("quality"))
    why_explanation = _build_why_explanation(why_payload)

    return LiveSensoryDiagnostics(
        symbol=resolved_symbol,
        generated_at=generated_at,
        samples=samples,
        integrated_signal=integrated_signal,
        anomaly=anomaly_payload,
        drift_summary=drift_summary,
        why_quality=why_quality,
        why_explanation=why_explanation,
        dimensions=dimensions,
    )


def build_live_sensory_diagnostics_from_manager(
    manager: "RealDataManager",
    *,
    symbol: str,
    period: str | None = None,
    interval: str | None = None,
    organ: RealSensoryOrgan | None = None,
    narrative_events: Sequence[NarrativeEvent] | None = None,
    macro_regime_flags: Mapping[str, float] | None = None,
) -> LiveSensoryDiagnostics:
    """Fetch live market data via ``RealDataManager`` and derive diagnostics."""

    frame = manager.fetch_data(symbol, period=period, interval=interval)
    if frame is None or frame.empty:
        raise ValueError("RealDataManager returned empty market data frame")

    return build_live_sensory_diagnostics(
        frame,
        symbol=symbol,
        organ=organ,
        narrative_events=narrative_events,
        macro_regime_flags=macro_regime_flags,
    )


def _resolve_symbol(symbol: str | None, frame: pd.DataFrame) -> str:
    if symbol:
        return str(symbol)
    if "symbol" in frame.columns and not frame.empty:
        value = frame["symbol"].iloc[-1]
        if isinstance(value, str):
            return value
    return "UNKNOWN"


def _ensure_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if value is None:
        return datetime.now(timezone.utc)
    converted = pd.to_datetime(value, utc=True, errors="coerce")
    if isinstance(converted, pd.Timestamp):
        return converted.to_pydatetime()
    if isinstance(converted, datetime):
        return converted if converted.tzinfo else converted.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _extract_dimensions(raw_dimensions: Any) -> dict[str, dict[str, Any]]:
    dimensions: dict[str, dict[str, Any]] = {}
    if not isinstance(raw_dimensions, Mapping):
        return dimensions

    for name, payload in raw_dimensions.items():
        if not isinstance(payload, Mapping):
            continue
        dimension_name = str(name)
        serialised: dict[str, Any] = {}
        for key in ("signal", "confidence", "value", "metadata", "quality", "lineage"):
            if key not in payload:
                continue
            value = payload.get(key)
            if key in {"metadata", "quality", "lineage"}:
                serialised[key] = _normalise_mapping(value)
            elif key == "value":
                serialised[key] = _normalise_value(value)
            else:
                serialised[key] = value
        dimensions[dimension_name] = serialised
    return dimensions


def _serialise_integrated_signal(value: Any) -> Mapping[str, Any]:
    if isinstance(value, IntegratedSignal):
        return {
            "direction": value.direction,
            "strength": value.strength,
            "confidence": value.confidence,
            "contributing": list(value.contributing),
        }
    if isinstance(value, Mapping):
        return dict(value)
    return {
        "direction": None,
        "strength": None,
        "confidence": None,
        "contributing": [],
    }


def _serialise_drift_summary(summary: Any) -> Mapping[str, Any] | None:
    if summary is None:
        return None
    if isinstance(summary, Mapping):
        return {str(key): value for key, value in summary.items()}
    if isinstance(summary, SensorDriftSummary):
        payload: MutableMapping[str, Any] = dict(summary.as_dict())
        payload["exceeded"] = [result.as_dict() for result in summary.exceeded]
        return payload
    return None


def _normalise_mapping(mapping: Any) -> dict[str, Any]:
    if not isinstance(mapping, Mapping):
        return {}
    return {str(key): _normalise_value(value) for key, value in mapping.items()}


def _normalise_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _normalise_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_normalise_value(item) for item in value]
    return value


def _build_why_explanation(why_payload: Mapping[str, Any]) -> str:
    metadata = _normalise_mapping(why_payload.get("metadata"))
    quality = _normalise_mapping(why_payload.get("quality"))
    value = _normalise_mapping(why_payload.get("value"))

    parts: list[str] = []

    strength = _coerce_float(value.get("strength"))
    confidence = _coerce_float(value.get("confidence"))
    if strength is not None and confidence is not None:
        parts.append(f"WHY strength {strength:+.3f} (confidence {confidence:.2f})")
    elif strength is not None:
        parts.append(f"WHY strength {strength:+.3f}")
    elif confidence is not None:
        parts.append(f"WHY confidence {confidence:.2f}")

    macro_bias = _coerce_float(metadata.get("macro_bias"))
    if macro_bias is not None:
        parts.append(f"macro bias {macro_bias:+.3f}")

    yield_curve = metadata.get("yield_curve")
    if isinstance(yield_curve, Mapping):
        direction = _coerce_float(yield_curve.get("direction"))
        if direction is not None:
            parts.append(f"yield direction {direction:+.3f}")
        slope_2s10s = _coerce_float(yield_curve.get("slope_2s10s"))
        if slope_2s10s is not None:
            parts.append(f"2s10s {slope_2s10s:+.3f}")

    data_quality = _coerce_float(quality.get("data_quality"))
    if data_quality is not None:
        parts.append(f"data quality {data_quality:.2f}")

    narrative = metadata.get("narrative")
    if isinstance(narrative, Mapping):
        theme = narrative.get("dominant_theme")
        if isinstance(theme, str) and theme:
            parts.append(f"theme {theme}")
        sentiment = _coerce_float(narrative.get("sentiment_score"))
        if sentiment is not None:
            parts.append(f"sentiment {sentiment:+.2f}")
        upcoming = narrative.get("upcoming_event")
        if isinstance(upcoming, Mapping):
            description = upcoming.get("description") or upcoming.get("name")
            minutes = _coerce_float(upcoming.get("minutes_ahead"))
            if isinstance(description, str) and minutes is not None:
                parts.append(f"next event {description} in {minutes:.0f} min")

    if not parts:
        return "WHY signal metadata unavailable"
    return "; ".join(parts)


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None
