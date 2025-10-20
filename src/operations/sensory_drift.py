"""Sensory drift telemetry helpers.

This module analyses recent sensory audit entries emitted by the
professional runtime and converts them into reusable telemetry snapshots.
The resulting payload captures dimension-level drift, severity, and
markdown summaries suitable for operator dashboards and event-bus feeds.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from math import inf, log
from statistics import StatisticsError, fmean, quantiles
from typing import Any, Iterable, Mapping, MutableMapping

import logging

from src.core.event_bus import Event, EventBus, TopicBus
from src.operations.alerts import AlertEvent, AlertSeverity
from src.operations.observability_diary import ThrottleStateSnapshot
from src.operations.event_bus_failover import publish_event_with_failover
from src.understanding.metrics import export_throttle_metrics


logger = logging.getLogger(__name__)


class DriftSeverity(str, Enum):
    """Severity levels exposed by the sensory drift snapshot."""

    normal = "normal"
    warn = "warn"
    alert = "alert"


_SEVERITY_ORDER: dict[DriftSeverity, int] = {
    DriftSeverity.normal: 0,
    DriftSeverity.warn: 1,
    DriftSeverity.alert: 2,
}


def _max_severity(current: DriftSeverity, candidate: DriftSeverity) -> DriftSeverity:
    if _SEVERITY_ORDER[candidate] > _SEVERITY_ORDER[current]:
        return candidate
    return current


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _sanitize_sequence(value: Iterable[Any]) -> list[Mapping[str, Any]]:
    cleaned: list[Mapping[str, Any]] = []
    for entry in value:
        if isinstance(entry, Mapping):
            cleaned.append(dict(entry))
    return cleaned


_PSI_ALERT_THRESHOLD = 0.25
_PSI_MIN_FEATURES = 8
_PSI_MAX_FEATURES = 12
_PSI_MIN_BASELINE_SAMPLES = 20
_PSI_MIN_EVALUATION_SAMPLES = 8
_PSI_MAX_BASELINE_SAMPLES = 120
_PSI_MAX_EVALUATION_SAMPLES = 24
_PSI_BUCKETS = 10
_PSI_EPSILON = 1e-6

_CORE_FEATURE_PRIORITY: tuple[str, ...] = (
    "unified_score",
    "confidence",
    "WHY.signal",
    "WHY.confidence",
    "WHAT.signal",
    "WHAT.confidence",
    "WHEN.signal",
    "WHEN.confidence",
    "HOW.signal",
    "HOW.confidence",
    "ANOMALY.signal",
    "ANOMALY.confidence",
)


def _iter_core_features(entry: Mapping[str, Any]) -> Iterable[tuple[str, float]]:
    unified_score = _coerce_float(entry.get("unified_score"))
    if unified_score is not None:
        yield "unified_score", unified_score

    overall_confidence = _coerce_float(entry.get("confidence"))
    if overall_confidence is not None:
        yield "confidence", overall_confidence

    dimensions = entry.get("dimensions")
    if not isinstance(dimensions, Mapping):
        return

    for name, payload in dimensions.items():
        if not isinstance(payload, Mapping):
            continue
        normalised_name = str(name).upper()
        signal_value = _coerce_float(payload.get("signal"))
        if signal_value is not None:
            yield f"{normalised_name}.signal", signal_value
        confidence_value = _coerce_float(payload.get("confidence"))
        if confidence_value is not None:
            yield f"{normalised_name}.confidence", confidence_value


def _aggregate_feature_samples(
    entries: Sequence[Mapping[str, Any]]
) -> Mapping[str, list[float]]:
    samples: dict[str, list[float]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        for feature_key, value in _iter_core_features(entry):
            samples.setdefault(feature_key, []).append(value)
    return samples


def _prepare_psi_windows(
    entries: Sequence[Mapping[str, Any]]
) -> tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]] | None:
    total_entries = len(entries)
    if total_entries < _PSI_MIN_BASELINE_SAMPLES + _PSI_MIN_EVALUATION_SAMPLES:
        return None

    max_evaluation = total_entries - _PSI_MIN_BASELINE_SAMPLES
    if max_evaluation < _PSI_MIN_EVALUATION_SAMPLES:
        return None

    evaluation_count = min(_PSI_MAX_EVALUATION_SAMPLES, max_evaluation)
    evaluation_count = max(_PSI_MIN_EVALUATION_SAMPLES, evaluation_count)

    baseline_slice = entries[evaluation_count : evaluation_count + _PSI_MAX_BASELINE_SAMPLES]
    if len(baseline_slice) < _PSI_MIN_BASELINE_SAMPLES:
        return None

    evaluation_slice = entries[:evaluation_count]
    return evaluation_slice, baseline_slice


def _build_psi_boundaries(values: Sequence[float]) -> list[float] | None:
    try:
        quantile_cuts = quantiles(values, n=_PSI_BUCKETS, method="inclusive")
    except StatisticsError:
        return None

    unique_cuts = sorted(set(quantile_cuts))
    if not unique_cuts:
        return None

    return [-inf, *unique_cuts, inf]


def _histogram(values: Sequence[float], boundaries: Sequence[float]) -> list[int]:
    counts = [0] * (len(boundaries) - 1)
    for value in values:
        index = bisect_right(boundaries, value) - 1
        if index < 0:
            index = 0
        elif index >= len(counts):
            index = len(counts) - 1
        counts[index] += 1
    return counts


def _calculate_psi(
    baseline_values: Sequence[float],
    evaluation_values: Sequence[float],
) -> float | None:
    if (
        len(baseline_values) < _PSI_MIN_BASELINE_SAMPLES
        or len(evaluation_values) < _PSI_MIN_EVALUATION_SAMPLES
    ):
        return None

    boundaries = _build_psi_boundaries(baseline_values)
    if boundaries is None:
        return None

    baseline_counts = _histogram(baseline_values, boundaries)
    evaluation_counts = _histogram(evaluation_values, boundaries)

    baseline_total = sum(baseline_counts)
    evaluation_total = sum(evaluation_counts)
    if baseline_total == 0 or evaluation_total == 0:
        return None

    psi_value = 0.0
    for base_count, eval_count in zip(baseline_counts, evaluation_counts):
        base_ratio = max(_PSI_EPSILON, base_count / baseline_total)
        eval_ratio = max(_PSI_EPSILON, eval_count / evaluation_total)
        psi_value += (eval_ratio - base_ratio) * log(eval_ratio / base_ratio)

    return psi_value


def _evaluate_population_stability(
    entries: Sequence[Mapping[str, Any]]
) -> tuple[Mapping[str, float], Mapping[str, object] | None]:
    window = _prepare_psi_windows(entries)
    if window is None:
        return {}, None

    evaluation_entries, baseline_entries = window
    baseline_samples = _aggregate_feature_samples(baseline_entries)
    evaluation_samples = _aggregate_feature_samples(evaluation_entries)

    feature_keys: list[str] = [
        key
        for key in _CORE_FEATURE_PRIORITY
        if key in baseline_samples and key in evaluation_samples
    ]

    if len(feature_keys) < _PSI_MIN_FEATURES:
        available = sorted(
            set(baseline_samples).intersection(evaluation_samples).difference(feature_keys)
        )
        for key in available:
            feature_keys.append(key)
            if len(feature_keys) >= _PSI_MIN_FEATURES:
                break

    if not feature_keys:
        return {}, None

    if len(feature_keys) > _PSI_MAX_FEATURES:
        feature_keys = feature_keys[:_PSI_MAX_FEATURES]

    psi_results: dict[str, float] = {}
    for key in feature_keys:
        baseline_values = baseline_samples.get(key, [])[:_PSI_MAX_BASELINE_SAMPLES]
        evaluation_values = evaluation_samples.get(key, [])[:_PSI_MAX_EVALUATION_SAMPLES]
        psi_value = _calculate_psi(baseline_values, evaluation_values)
        if psi_value is not None:
            psi_results[key] = psi_value

    if not psi_results:
        return {}, None

    metadata: dict[str, object] = {
        "threshold": _PSI_ALERT_THRESHOLD,
        "feature_count": len(psi_results),
        "features": [
            {"name": key, "psi": psi_results[key]}
            for key in sorted(psi_results)
        ],
        "baseline_samples": len(baseline_entries),
        "evaluation_samples": len(evaluation_entries),
        "max_psi": max(psi_results.values()),
    }

    alert_features = [
        {"name": key, "psi": value}
        for key, value in psi_results.items()
        if value >= _PSI_ALERT_THRESHOLD
    ]
    if alert_features:
        metadata["alerts"] = sorted(alert_features, key=lambda item: item["psi"], reverse=True)

    return psi_results, metadata


def _page_hinkley_stat(values: Sequence[float], *, delta: float) -> float:
    if not values:
        return 0.0
    mean = values[0]
    positive_sum = 0.0
    negative_sum = 0.0
    positive_max = 0.0
    negative_max = 0.0
    for index, value in enumerate(values[1:], start=1):
        mean += (value - mean) / (index + 1)
        deviation = value - mean - delta
        positive_sum = max(0.0, positive_sum + deviation)
        negative_sum = min(0.0, negative_sum + deviation)
        positive_max = max(positive_max, positive_sum)
        negative_max = min(negative_max, negative_sum)

    return max(positive_max, abs(negative_max))


def _variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = fmean(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _variance_ratio(
    baseline: Sequence[float], evaluation: Sequence[float]
) -> float | None:
    if len(baseline) < 2 or len(evaluation) < 2:
        return None
    baseline_var = _variance(baseline)
    evaluation_var = _variance(evaluation)
    if baseline_var <= 0.0:
        return inf if evaluation_var > 0.0 else None
    return evaluation_var / baseline_var


def _upgrade_severity(current: DriftSeverity, candidate: DriftSeverity) -> DriftSeverity:
    if _SEVERITY_ORDER[candidate] > _SEVERITY_ORDER[current]:
        return candidate
    return current


@dataclass(frozen=True)
class SensoryDimensionDrift:
    """Drift statistics for a single sensory dimension."""

    name: str
    current_signal: float
    baseline_signal: float | None
    delta: float | None
    current_confidence: float | None
    baseline_confidence: float | None
    confidence_delta: float | None
    severity: DriftSeverity
    samples: int
    page_hinkley_stat: float | None = None
    variance_ratio: float | None = None
    detectors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "current_signal": self.current_signal,
            "severity": self.severity.value,
            "samples": self.samples,
        }
        if self.baseline_signal is not None:
            payload["baseline_signal"] = self.baseline_signal
        if self.delta is not None:
            payload["delta"] = self.delta
        if self.current_confidence is not None:
            payload["current_confidence"] = self.current_confidence
        if self.baseline_confidence is not None:
            payload["baseline_confidence"] = self.baseline_confidence
        if self.confidence_delta is not None:
            payload["confidence_delta"] = self.confidence_delta
        if self.page_hinkley_stat is not None:
            payload["page_hinkley_stat"] = self.page_hinkley_stat
        if self.variance_ratio is not None:
            payload["variance_ratio"] = self.variance_ratio
        if self.detectors:
            payload["detectors"] = list(self.detectors)
        return payload


@dataclass(frozen=True)
class SensoryDriftSnapshot:
    """Aggregate drift telemetry across all sensory dimensions."""

    generated_at: datetime
    status: DriftSeverity
    dimensions: Mapping[str, SensoryDimensionDrift]
    sample_window: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "status": self.status.value,
            "sample_window": self.sample_window,
            "metadata": dict(self.metadata),
            "dimensions": {
                name: dimension.as_dict() for name, dimension in self.dimensions.items()
            },
        }

    def to_markdown(self) -> str:
        if not self.dimensions:
            return "No sensory audit data available."

        header = "| Dimension | Signal Δ | Confidence Δ | Status | Samples |"
        separator = "| --- | --- | --- | --- | --- |"
        rows: list[str] = [header, separator]

        for name in sorted(self.dimensions):
            dimension = self.dimensions[name]
            delta = dimension.delta if dimension.delta is not None else 0.0
            confidence_delta = (
                dimension.confidence_delta if dimension.confidence_delta is not None else 0.0
            )
            rows.append(
                f"| {name} | {delta:+.3f} | {confidence_delta:+.3f} | {dimension.severity.value} | {dimension.samples} |"
            )

        return "\n".join(rows)


def evaluate_sensory_drift(
    audit_entries: Sequence[Mapping[str, Any]],
    *,
    lookback: int = 20,
    warn_threshold: float = 0.25,
    alert_threshold: float = 0.5,
    page_hinkley_delta: float = 0.01,
    page_hinkley_warn: float = 3.0,
    page_hinkley_alert: float = 6.0,
    variance_window: int | None = None,
    variance_warn_ratio: float = 1.5,
    variance_alert_ratio: float = 2.5,
    min_variance_samples: int = 5,
    metadata: Mapping[str, Any] | None = None,
) -> SensoryDriftSnapshot:
    """Evaluate sensor drift from recent audit entries.

    The detector combines simple delta thresholds with Page–Hinkley drift
    statistics and rolling variance ratios to emulate the DriftSentry brief.
    ``page_hinkley_*`` parameters control the cumulative mean shift detector,
    while the variance options guard volatility spikes in the trailing window.
    ``min_variance_samples`` ensures both baseline and evaluation slices have
    sufficient observations before a variance ratio is emitted.
    """

    cleaned_entries = _sanitize_sequence(audit_entries)
    generated_at = datetime.utcnow()
    if not cleaned_entries:
        snapshot = SensoryDriftSnapshot(
            generated_at=generated_at,
            status=DriftSeverity.normal,
            dimensions={},
            sample_window=0,
            metadata={"reason": "no_audit_entries"} | (dict(metadata) if metadata else {}),
        )
        return snapshot

    latest = cleaned_entries[0]
    history = cleaned_entries[1 : lookback + 1]

    latest_dimensions = latest.get("dimensions")
    if not isinstance(latest_dimensions, Mapping):
        latest_dimensions = {}

    dimension_payloads: MutableMapping[str, SensoryDimensionDrift] = {}
    aggregate_status = DriftSeverity.normal
    detector_catalog: dict[str, dict[str, Any]] = {}
    severity_counts: Counter[str] = Counter()

    for name, payload in latest_dimensions.items():
        if not isinstance(payload, Mapping):
            continue

        current_signal = _coerce_float(payload.get("signal"))
        if current_signal is None:
            continue

        current_confidence = _coerce_float(payload.get("confidence"))

        baseline_signals: list[float] = []
        baseline_confidences: list[float] = []

        for entry in history:
            entry_dims = entry.get("dimensions")
            if not isinstance(entry_dims, Mapping):
                continue
            historic_payload = entry_dims.get(name)
            if not isinstance(historic_payload, Mapping):
                continue
            baseline_signal = _coerce_float(historic_payload.get("signal"))
            if baseline_signal is not None:
                baseline_signals.append(baseline_signal)
            baseline_confidence = _coerce_float(historic_payload.get("confidence"))
            if baseline_confidence is not None:
                baseline_confidences.append(baseline_confidence)

        baseline_signal_value = fmean(baseline_signals) if baseline_signals else None
        baseline_confidence_value = fmean(baseline_confidences) if baseline_confidences else None

        delta: float | None = None
        severity = DriftSeverity.normal
        detectors: list[str] = []

        if baseline_signal_value is not None:
            delta = current_signal - baseline_signal_value
            absolute_delta = abs(delta)
            if absolute_delta >= alert_threshold:
                severity = DriftSeverity.alert
                detectors.append("delta_alert")
            elif absolute_delta >= warn_threshold:
                severity = DriftSeverity.warn
                detectors.append("delta_warn")

        detector_series = list(reversed(baseline_signals)) + [current_signal]
        page_hinkley_stat: float | None = None
        variance_ratio: float | None = None

        if len(detector_series) >= 4:
            page_hinkley_stat = _page_hinkley_stat(detector_series, delta=page_hinkley_delta)
            if page_hinkley_stat >= page_hinkley_alert:
                severity = DriftSeverity.alert
                detectors.append("page_hinkley_alert")
            elif page_hinkley_stat >= page_hinkley_warn:
                severity = _upgrade_severity(severity, DriftSeverity.warn)
                detectors.append("page_hinkley_warn")

        min_total_variance = 2 * min_variance_samples
        series_length = len(detector_series)
        if series_length >= min_total_variance:
            eval_window = variance_window if variance_window is not None else lookback
            if eval_window <= 0:
                eval_window = min_variance_samples
            eval_window = max(min_variance_samples, min(eval_window, series_length - min_variance_samples))
            if eval_window >= min_variance_samples and series_length - eval_window >= min_variance_samples:
                evaluation_values = detector_series[-eval_window:]
                baseline_window_values = detector_series[: series_length - eval_window]
                baseline_values = baseline_window_values[-eval_window:]
                ratio = _variance_ratio(baseline_values, evaluation_values)
                if ratio is not None:
                    variance_ratio = ratio
                    if ratio >= variance_alert_ratio:
                        severity = DriftSeverity.alert
                        detectors.append("variance_alert")
                    elif ratio >= variance_warn_ratio:
                        severity = _upgrade_severity(severity, DriftSeverity.warn)
                        detectors.append("variance_warn")

        aggregate_status = _max_severity(aggregate_status, severity)

        confidence_delta: float | None = None
        if baseline_confidence_value is not None and current_confidence is not None:
            confidence_delta = current_confidence - baseline_confidence_value

        dimension_payloads[name] = SensoryDimensionDrift(
            name=name,
            current_signal=current_signal,
            baseline_signal=baseline_signal_value,
            delta=delta,
            current_confidence=current_confidence,
            baseline_confidence=baseline_confidence_value,
            confidence_delta=confidence_delta,
            severity=severity,
            samples=1 + len(baseline_signals),
            page_hinkley_stat=page_hinkley_stat,
            variance_ratio=variance_ratio,
            detectors=tuple(detectors),
        )
        severity_counts[severity.value] += 1
        detector_entry: dict[str, Any] = {"severity": severity.value}
        if detectors:
            detector_entry["detectors"] = list(detectors)
        if page_hinkley_stat is not None:
            detector_entry["page_hinkley_stat"] = page_hinkley_stat
        if variance_ratio is not None:
            detector_entry["variance_ratio"] = variance_ratio
        detector_catalog[name] = detector_entry

    psi_results, psi_metadata = _evaluate_population_stability(cleaned_entries)

    if psi_results:
        psi_baseline_samples = int(
            psi_metadata.get("baseline_samples", 0) if psi_metadata else 0
        )
        psi_evaluation_samples = int(
            psi_metadata.get("evaluation_samples", 0) if psi_metadata else 0
        )
        psi_total_samples = max(psi_baseline_samples + psi_evaluation_samples, 0)

        for feature_name, psi_value in psi_results.items():
            severity = (
                DriftSeverity.alert
                if psi_value >= _PSI_ALERT_THRESHOLD
                else DriftSeverity.normal
            )
            detectors = ["psi"]
            if severity is DriftSeverity.alert:
                detectors.append("psi_alert")

            dimension_name = f"psi:{feature_name}"
            dimension_payloads[dimension_name] = SensoryDimensionDrift(
                name=dimension_name,
                current_signal=psi_value,
                baseline_signal=None,
                delta=None,
                current_confidence=None,
                baseline_confidence=None,
                confidence_delta=None,
                severity=severity,
                samples=psi_total_samples,
                detectors=tuple(detectors),
            )
            severity_counts[severity.value] += 1
            detector_entry = {
                "severity": severity.value,
                "psi": psi_value,
                "threshold": _PSI_ALERT_THRESHOLD,
            }
            if detectors:
                detector_entry["detectors"] = list(detectors)
            detector_catalog[dimension_name] = detector_entry
            aggregate_status = _max_severity(aggregate_status, severity)

    snapshot_metadata = {"entries": len(cleaned_entries)}
    if metadata:
        snapshot_metadata.update(dict(metadata))
    if severity_counts:
        snapshot_metadata["severity_counts"] = dict(severity_counts)
    if detector_catalog:
        snapshot_metadata["detectors"] = detector_catalog
    if psi_metadata:
        snapshot_metadata["psi"] = psi_metadata

    sample_window = min(len(cleaned_entries), lookback + 1)
    if psi_metadata:
        psi_window = int(
            (psi_metadata.get("baseline_samples") or 0)
            + (psi_metadata.get("evaluation_samples") or 0)
        )
        if psi_window > 0:
            sample_window = max(sample_window, min(len(cleaned_entries), psi_window))

    return SensoryDriftSnapshot(
        generated_at=generated_at,
        status=aggregate_status,
        dimensions=dimension_payloads,
        sample_window=sample_window,
        metadata=snapshot_metadata,
    )


def publish_sensory_drift(
    event_bus: EventBus,
    snapshot: SensoryDriftSnapshot,
    *,
    global_bus_factory: Callable[[], TopicBus] | None = None,
) -> None:
    """Publish the sensory drift snapshot to the runtime event bus."""

    event = Event(
        type="telemetry.sensory.drift",
        payload=snapshot.as_dict(),
        source="operations.sensory_drift",
    )

    publish_event_with_failover(
        event_bus,
        event,
        logger=logger,
        runtime_fallback_message=
            "Primary event bus publish_from_sync failed; falling back to global bus",
        runtime_unexpected_message=
            "Unexpected error publishing sensory drift telemetry via runtime bus",
        runtime_none_message=
            "Primary event bus publish_from_sync returned None; falling back to global bus",
        global_not_running_message=
            "Global event bus not running while publishing sensory drift telemetry",
        global_unexpected_message=
            "Unexpected error publishing sensory drift telemetry via global bus",
        global_bus_factory=global_bus_factory,
    )


def _should_emit(severity: DriftSeverity, threshold: DriftSeverity) -> bool:
    return _SEVERITY_ORDER[severity] >= _SEVERITY_ORDER[threshold]


def derive_drift_alerts(
    snapshot: SensoryDriftSnapshot,
    *,
    threshold: DriftSeverity = DriftSeverity.warn,
    include_overall: bool = True,
    base_tags: Sequence[str] = ("drift-sentry",),
) -> list[AlertEvent]:
    """Translate sensory drift telemetry into alert events."""

    events: list[AlertEvent] = []
    base_tag_tuple = tuple(base_tags)
    severity_map: Mapping[DriftSeverity, AlertSeverity] = {
        DriftSeverity.normal: AlertSeverity.info,
        DriftSeverity.warn: AlertSeverity.warning,
        DriftSeverity.alert: AlertSeverity.critical,
    }

    if include_overall and _should_emit(snapshot.status, threshold):
        events.append(
            AlertEvent(
                category="sensory.drift",
                severity=severity_map[snapshot.status],
                message=f"Sensory drift status {snapshot.status.value}",
                tags=base_tag_tuple,
                context={"snapshot": snapshot.as_dict()},
            )
        )

    for name, dimension in snapshot.dimensions.items():
        if not _should_emit(dimension.severity, threshold):
            continue
        dimension_tags = base_tag_tuple + (name.lower(),)
        detector_suffix = f" ({', '.join(dimension.detectors)})" if dimension.detectors else ""
        events.append(
            AlertEvent(
                category=f"sensory.drift.{name.lower()}",
                severity=severity_map[dimension.severity],
                message=f"{name} drift {dimension.severity.value}{detector_suffix}",
                tags=dimension_tags,
                context={
                    "dimension": dimension.as_dict(),
                    "snapshot": snapshot.as_dict(),
                },
            )
        )

    return events


def export_drift_throttle_metrics(
    snapshot: SensoryDriftSnapshot,
    throttle_states: Iterable[ThrottleStateSnapshot | Mapping[str, Any]],
    *,
    regime: str | None = None,
    decision_id: str | None = None,
    threshold: DriftSeverity = DriftSeverity.warn,
) -> bool:
    """Export throttle posture to Prometheus when drift escalates."""

    states = tuple(throttle_states)
    if not states:
        return False

    should_export = _should_emit(snapshot.status, threshold)
    if not should_export:
        should_export = any(
            _should_emit(dimension.severity, threshold)
            for dimension in snapshot.dimensions.values()
        )

    if not should_export:
        return False

    export_throttle_metrics(states, regime=regime, decision_id=decision_id)
    return True


__all__ = [
    "DriftSeverity",
    "SensoryDimensionDrift",
    "SensoryDriftSnapshot",
    "evaluate_sensory_drift",
    "derive_drift_alerts",
    "publish_sensory_drift",
    "export_drift_throttle_metrics",
]
