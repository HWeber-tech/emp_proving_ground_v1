"""Continuous calibration and lineage quality instrumentation for sensors.

The roadmap calls for the sensory cortex to adjust itself continuously while
guarding lineage metadata quality for real-time governance.  This module keeps
the runtime cost low by maintaining rolling windows of sensor strengths per
dimension, recalibrating thresholds on a cadence, and emitting summaries that
can be inspected or published alongside sensory telemetry.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import numpy as np

CalibrationCallback = Callable[["CalibrationUpdate"], None]


@dataclass(slots=True)
class ContinuousCalibrationConfig:
    """Global configuration for the continuous calibration harness."""

    window: int = 512
    min_samples: int = 40
    recalibration_interval: int = 10
    warn_quantile: float = 0.75
    alert_quantile: float = 0.9
    lineage_staleness: timedelta = timedelta(minutes=5)
    required_lineage_keys: tuple[str, ...] = (
        "dimension",
        "source",
        "generated_at",
        "inputs",
        "outputs",
    )
    quality_required_keys: tuple[str, ...] = (
        "timestamp",
        "confidence",
        "strength",
    )


@dataclass(slots=True)
class CalibrationUpdate:
    """Snapshot describing a recalibration event for a single dimension."""

    dimension: str
    samples: int
    warn_threshold: float | None
    alert_threshold: float | None
    mean_strength: float
    std_strength: float
    median_strength: float
    percentile_75: float | None
    percentile_90: float | None
    absolute_mode: bool
    lineage_issues: tuple[str, ...]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def as_dict(self) -> Mapping[str, Any]:
        """Serialise the calibration update for telemetry surfaces."""

        payload: MutableMapping[str, Any] = {
            "dimension": self.dimension,
            "samples": self.samples,
            "warn_threshold": self.warn_threshold,
            "alert_threshold": self.alert_threshold,
            "mean_strength": self.mean_strength,
            "std_strength": self.std_strength,
            "median_strength": self.median_strength,
            "percentile_75": self.percentile_75,
            "percentile_90": self.percentile_90,
            "absolute_mode": self.absolute_mode,
            "generated_at": self.generated_at.isoformat(),
        }
        if self.lineage_issues:
            payload["lineage_issues"] = list(self.lineage_issues)
        return payload


@dataclass(slots=True)
class _DimensionRegistration:
    dimension: str
    apply_callback: CalibrationCallback | None
    calibrate: bool
    absolute_strength: bool
    clamp_min: float
    clamp_max: float
    warn_quantile: float
    alert_quantile: float


@dataclass(slots=True)
class _DimensionState:
    config: _DimensionRegistration
    values: deque[float] = field(default_factory=deque)
    samples: int = 0
    since_last_update: int = 0
    last_update: CalibrationUpdate | None = None
    issues: tuple[str, ...] = ()
    last_issue_at: datetime | None = None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_mapping(payload: Any) -> Mapping[str, Any] | None:
    if isinstance(payload, Mapping):
        return payload
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str):
        candidate = value.strip()
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            timestamp = datetime.fromisoformat(candidate)
        except ValueError:
            return None
    else:
        return None

    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


class LineageQualityChecker:
    """Assess lineage payloads for completeness and freshness."""

    def __init__(
        self,
        *,
        required_keys: Sequence[str],
        staleness: timedelta,
        quality_required_keys: Sequence[str],
    ) -> None:
        self._required_keys = tuple(required_keys)
        self._staleness = staleness
        self._quality_required_keys = tuple(quality_required_keys)

    def assess(
        self,
        dimension: str,
        lineage: Any,
        *,
        quality_payload: Mapping[str, Any] | None,
    ) -> tuple[str, ...]:
        issues: list[str] = []
        payload = self._normalise_lineage(lineage)
        if payload is None:
            issues.append("missing_lineage_payload")
        else:
            for key in self._required_keys:
                if key not in payload:
                    issues.append(f"lineage_missing_{key}")
            lineage_dimension = payload.get("dimension")
            if isinstance(lineage_dimension, str) and lineage_dimension != dimension:
                issues.append("lineage_dimension_mismatch")

            generated_at = payload.get("generated_at")
            timestamp = _parse_timestamp(generated_at)
            if timestamp is None:
                issues.append("lineage_generated_at_invalid")
            else:
                if datetime.now(timezone.utc) - timestamp > self._staleness:
                    issues.append("lineage_stale")

        quality = _coerce_mapping(quality_payload)
        if not quality:
            issues.append("quality_metadata_missing")
        else:
            for key in self._quality_required_keys:
                if key not in quality:
                    issues.append(f"quality_missing_{key}")
            timestamp_value = _parse_timestamp(quality.get("timestamp")) if quality else None
            if timestamp_value is None:
                issues.append("quality_timestamp_invalid")
            elif datetime.now(timezone.utc) - timestamp_value > self._staleness:
                issues.append("quality_timestamp_stale")

        # Deduplicate while preserving order of first appearance.
        deduped: list[str] = []
        for issue in issues:
            if issue not in deduped:
                deduped.append(issue)
        return tuple(deduped)

    def _normalise_lineage(self, lineage: Any) -> Mapping[str, Any] | None:
        if lineage is None:
            return None
        if hasattr(lineage, "as_dict") and callable(lineage.as_dict):  # type: ignore[attr-defined]
            try:
                value = lineage.as_dict()  # type: ignore[no-any-call]
                return value if isinstance(value, Mapping) else None
            except Exception:
                return None
        if isinstance(lineage, Mapping):
            return lineage
        return None


class ContinuousSensorCalibrator:
    """Maintain sliding calibration windows and lineage quality checks."""

    def __init__(
        self,
        config: ContinuousCalibrationConfig | None = None,
        *,
        lineage_checker: LineageQualityChecker | None = None,
    ) -> None:
        self._config = config or ContinuousCalibrationConfig()
        self._lineage_checker = lineage_checker or LineageQualityChecker(
            required_keys=self._config.required_lineage_keys,
            staleness=self._config.lineage_staleness,
            quality_required_keys=self._config.quality_required_keys,
        )
        self._dimensions: dict[str, _DimensionState] = {}

    # ------------------------------------------------------------------
    def register_dimension(
        self,
        dimension: str,
        *,
        apply_callback: CalibrationCallback | None = None,
        calibrate: bool | None = None,
        absolute_strength: bool = True,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
        warn_quantile: float | None = None,
        alert_quantile: float | None = None,
    ) -> None:
        """Register a dimension for calibration and lineage tracking."""

        warn_q = warn_quantile if warn_quantile is not None else self._config.warn_quantile
        alert_q = alert_quantile if alert_quantile is not None else self._config.alert_quantile
        should_calibrate = bool(apply_callback) if calibrate is None else calibrate

        registration = _DimensionRegistration(
            dimension=dimension,
            apply_callback=apply_callback,
            calibrate=should_calibrate,
            absolute_strength=absolute_strength,
            clamp_min=float(clamp_min),
            clamp_max=float(clamp_max),
            warn_quantile=warn_q,
            alert_quantile=alert_q,
        )
        state = _DimensionState(
            config=registration,
            values=deque(maxlen=self._config.window),
        )
        self._dimensions[dimension] = state

    # ------------------------------------------------------------------
    def observe(
        self,
        dimension: str,
        payload: Mapping[str, Any] | None,
        lineage: Any,
    ) -> None:
        """Record a sensory payload for calibration and lineage checks."""

        state = self._dimensions.get(dimension)
        if state is None:
            self.register_dimension(dimension, calibrate=False)
            state = self._dimensions[dimension]

        strength = self._extract_strength(payload)
        if strength is not None:
            measurement = abs(strength) if state.config.absolute_strength else strength
            state.values.append(float(measurement))
            state.samples += 1
            state.since_last_update += 1

        quality_payload: Mapping[str, Any] | None = None
        if isinstance(payload, Mapping):
            quality_payload = self._resolve_quality(payload)
        issues = self._lineage_checker.assess(dimension, lineage, quality_payload=quality_payload)
        if issues:
            state.issues = issues
            state.last_issue_at = datetime.now(timezone.utc)
        else:
            state.issues = ()

        if not state.config.calibrate:
            return

        if len(state.values) < self._config.min_samples:
            return

        if state.since_last_update < self._config.recalibration_interval:
            return

        update = self._build_update(dimension, state)
        state.last_update = update
        state.since_last_update = 0

        if state.config.apply_callback is not None:
            try:
                state.config.apply_callback(update)
            except Exception:
                # Calibration should never stop the sensory loop; issues surface via status.
                pass

    # ------------------------------------------------------------------
    def status(self) -> Mapping[str, Any]:
        """Expose calibration state for telemetry and diagnostics."""

        dimensions: dict[str, Mapping[str, Any]] = {}
        aggregated_issues: list[str] = []
        for dimension, state in sorted(self._dimensions.items()):
            latest = state.last_update.as_dict() if state.last_update else None
            issues = list(state.issues)
            aggregated_issues.extend(f"{dimension}:{issue}" for issue in issues)
            dimensions[dimension] = {
                "samples": state.samples,
                "window": len(state.values),
                "last_update": latest,
                "issues": issues,
                "last_issue_at": state.last_issue_at.isoformat() if state.last_issue_at else None,
            }

        return {
            "dimensions": dimensions,
            "issues": aggregated_issues,
        }

    # ------------------------------------------------------------------
    def _build_update(self, dimension: str, state: _DimensionState) -> CalibrationUpdate:
        values = np.array(state.values, dtype=float)
        mean_strength = float(np.mean(values)) if values.size else 0.0
        std_strength = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        median_strength = float(np.median(values)) if values.size else 0.0

        warn_threshold: float | None = None
        alert_threshold: float | None = None
        percentile_75: float | None = None
        percentile_90: float | None = None

        if state.config.calibrate and values.size:
            percentile_75 = float(np.quantile(values, state.config.warn_quantile))
            percentile_90 = float(np.quantile(values, state.config.alert_quantile))

            warn_threshold = max(state.config.clamp_min, min(state.config.clamp_max, percentile_75))
            alert_threshold = max(state.config.clamp_min, min(state.config.clamp_max, percentile_90))

            if alert_threshold < warn_threshold:
                alert_threshold = min(state.config.clamp_max, warn_threshold + 1e-6)

        return CalibrationUpdate(
            dimension=dimension,
            samples=state.samples,
            warn_threshold=warn_threshold,
            alert_threshold=alert_threshold,
            mean_strength=mean_strength,
            std_strength=std_strength,
            median_strength=median_strength,
            percentile_75=percentile_75,
            percentile_90=percentile_90,
            absolute_mode=state.config.absolute_strength,
            lineage_issues=state.issues,
        )

    def _extract_strength(self, payload: Mapping[str, Any] | None) -> float | None:
        if not payload:
            return None
        strength = payload.get("signal")
        if strength is None and isinstance(payload.get("value"), Mapping):
            strength = payload["value"].get("strength")
        return _coerce_float(strength)

    def _resolve_quality(self, payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
        quality = payload.get("quality")
        if isinstance(quality, Mapping):
            return quality
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            quality = metadata.get("quality")
            if isinstance(quality, Mapping):
                return quality
        return None


__all__ = [
    "CalibrationUpdate",
    "ContinuousCalibrationConfig",
    "ContinuousSensorCalibrator",
    "LineageQualityChecker",
]

