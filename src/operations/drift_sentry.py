"""Drift sentry detectors for belief/regime metrics.

This module implements the roadmap deliverable for the AlphaTrade
understanding loop sprint by combining Page–Hinkley and rolling variance
detectors into a reusable snapshot surface.  The resulting telemetry aligns
with the existing drift sentry gate, emits alert-ready payloads, and feeds the
operational readiness dashboard with severity-aware metadata so responders can
triage belief/regime drift using the shared runbook.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import inf
from statistics import fmean
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

from src.core.event_bus import Event, EventBus, TopicBus
from src.operations.alerts import AlertEvent, AlertSeverity
from src.operations.event_bus_failover import publish_event_with_failover
from src.operations.sensory_drift import DriftSeverity

import logging


logger = logging.getLogger(__name__)


def _coerce_float_sequence(values: Iterable[object]) -> list[float]:
    cleaned: list[float] = []
    for value in values:
        if isinstance(value, (int, float)):
            cleaned.append(float(value))
            continue
        try:
            cleaned.append(float(str(value).strip()))
        except (TypeError, ValueError):
            continue
    return cleaned


def _variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = fmean(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _page_hinkley_stat(values: Sequence[float], *, delta: float) -> float | None:
    if len(values) < 2:
        return None
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


def _cusum_stat(
    values: Sequence[float], *, reference: float | None, drift: float = 0.0
) -> float | None:
    if len(values) < 2 or reference is None:
        return None
    positive_sum = 0.0
    negative_sum = 0.0
    positive_max = 0.0
    negative_max = 0.0
    for value in values:
        deviation = value - reference - drift
        positive_sum = max(0.0, positive_sum + deviation)
        negative_sum = min(0.0, negative_sum + deviation)
        positive_max = max(positive_max, positive_sum)
        negative_max = min(negative_max, negative_sum)
    return max(positive_max, abs(negative_max))


_SEVERITY_ORDER: Mapping[DriftSeverity, int] = {
    DriftSeverity.normal: 0,
    DriftSeverity.warn: 1,
    DriftSeverity.alert: 2,
}


def _max_severity(first: DriftSeverity, second: DriftSeverity) -> DriftSeverity:
    return first if _SEVERITY_ORDER[first] >= _SEVERITY_ORDER[second] else second


@dataclass(slots=True, frozen=True)
class DriftSentryConfig:
    """Configuration parameters for understanding-loop drift detection."""

    baseline_window: int = 240
    evaluation_window: int = 60
    min_observations: int = 30
    page_hinkley_delta: float = 0.05
    page_hinkley_warn: float = 2.0
    page_hinkley_alert: float = 4.0
    cusum_drift: float = 0.0
    cusum_warn: float = 3.0
    cusum_alert: float = 6.0
    variance_ratio_warn: float = 1.6
    variance_ratio_alert: float = 2.4

    def as_dict(self) -> Mapping[str, float | int]:
        return {
            "baseline_window": self.baseline_window,
            "evaluation_window": self.evaluation_window,
            "min_observations": self.min_observations,
            "page_hinkley_delta": self.page_hinkley_delta,
            "page_hinkley_warn": self.page_hinkley_warn,
            "page_hinkley_alert": self.page_hinkley_alert,
            "cusum_drift": self.cusum_drift,
            "cusum_warn": self.cusum_warn,
            "cusum_alert": self.cusum_alert,
            "variance_ratio_warn": self.variance_ratio_warn,
            "variance_ratio_alert": self.variance_ratio_alert,
        }


@dataclass(slots=True, frozen=True)
class DriftSentryMetric:
    """Detector output for a single belief/regime metric."""

    name: str
    severity: DriftSeverity
    baseline_mean: float
    evaluation_mean: float
    baseline_variance: float
    evaluation_variance: float
    baseline_count: int
    evaluation_count: int
    page_hinkley_stat: float | None
    cusum_stat: float | None
    variance_ratio: float | None
    detectors: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "name": self.name,
            "severity": self.severity.value,
            "baseline_mean": self.baseline_mean,
            "evaluation_mean": self.evaluation_mean,
            "baseline_variance": self.baseline_variance,
            "evaluation_variance": self.evaluation_variance,
            "baseline_count": self.baseline_count,
            "evaluation_count": self.evaluation_count,
        }
        if self.page_hinkley_stat is not None:
            payload["page_hinkley_stat"] = self.page_hinkley_stat
        if self.cusum_stat is not None:
            payload["cusum_stat"] = self.cusum_stat
        if self.variance_ratio is not None:
            payload["variance_ratio"] = self.variance_ratio
        if self.detectors:
            payload["detectors"] = list(self.detectors)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True, frozen=True)
class DriftSentrySnapshot:
    """Aggregated drift sentry snapshot for operational readiness."""

    generated_at: datetime
    status: DriftSeverity
    metrics: Mapping[str, DriftSentryMetric]
    config: DriftSentryConfig
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, object]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "status": self.status.value,
            "metrics": {name: metric.as_dict() for name, metric in self.metrics.items()},
            "config": dict(self.config.as_dict()),
            "metadata": dict(self.metadata),
        }

    def to_markdown(self) -> str:
        if not self.metrics:
            return "No drift sentry metrics available."
        lines = [
            f"**Drift sentry status:** {self.status.value}",
            f"Generated at: {self.generated_at.isoformat()}",
            "",
            "| Metric | Severity | Δ Mean | Variance ratio | Detectors |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name, metric in sorted(self.metrics.items()):
            delta_mean = metric.evaluation_mean - metric.baseline_mean
            variance_ratio = (
                f"{metric.variance_ratio:.3f}" if metric.variance_ratio is not None else "n/a"
            )
            detectors = ", ".join(metric.detectors) if metric.detectors else "—"
            lines.append(
                f"| {name} | {metric.severity.value} | {delta_mean:+.3f} | {variance_ratio} | {detectors} |"
            )
        return "\n".join(lines)


def _severity_from_thresholds(
    value: float | None,
    warn_threshold: float,
    alert_threshold: float,
) -> DriftSeverity:
    if value is None:
        return DriftSeverity.normal
    if value >= alert_threshold:
        return DriftSeverity.alert
    if value >= warn_threshold:
        return DriftSeverity.warn
    return DriftSeverity.normal


def evaluate_drift_sentry(
    metrics: Mapping[str, Sequence[object]],
    *,
    config: DriftSentryConfig | None = None,
    generated_at: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
) -> DriftSentrySnapshot:
    """Evaluate belief/regime drift using Page–Hinkley and variance detectors."""

    cfg = config or DriftSentryConfig()
    generated = generated_at or datetime.now(tz=UTC)
    status = DriftSeverity.normal
    metric_payloads: dict[str, DriftSentryMetric] = {}
    severity_counts: Counter[str] = Counter()
    detector_catalog: dict[str, dict[str, object]] = {}

    window = cfg.baseline_window + cfg.evaluation_window
    if window <= 0:
        raise ValueError("baseline_window and evaluation_window must be positive")

    for name, raw_series in metrics.items():
        cleaned = _coerce_float_sequence(raw_series)
        if len(cleaned) < window:
            continue
        baseline = cleaned[-window:-cfg.evaluation_window]
        evaluation = cleaned[-cfg.evaluation_window :]
        if len(baseline) < cfg.min_observations or len(evaluation) < cfg.min_observations:
            continue

        combined_sequence = baseline + evaluation
        page_stat = _page_hinkley_stat(combined_sequence, delta=cfg.page_hinkley_delta)
        baseline_variance = _variance(baseline)
        evaluation_variance = _variance(evaluation)
        cusum_stat = _cusum_stat(
            evaluation,
            reference=fmean(baseline) if baseline else None,
            drift=cfg.cusum_drift,
        )
        if baseline_variance <= 0.0:
            variance_ratio: float | None = (
                inf if evaluation_variance > 0.0 else None
            )
        else:
            variance_ratio = evaluation_variance / baseline_variance

        baseline_mean = fmean(baseline)
        evaluation_mean = fmean(evaluation)

        page_severity = _severity_from_thresholds(
            page_stat,
            cfg.page_hinkley_warn,
            cfg.page_hinkley_alert,
        )
        cusum_severity = _severity_from_thresholds(
            cusum_stat,
            cfg.cusum_warn,
            cfg.cusum_alert,
        )
        variance_severity = _severity_from_thresholds(
            variance_ratio,
            cfg.variance_ratio_warn,
            cfg.variance_ratio_alert,
        )
        metric_severity = _max_severity(
            _max_severity(page_severity, cusum_severity),
            variance_severity,
        )

        detectors: list[str] = []
        if page_severity is DriftSeverity.warn:
            detectors.append("page_hinkley_warn")
        elif page_severity is DriftSeverity.alert:
            detectors.append("page_hinkley_alert")
        if cusum_severity is DriftSeverity.warn:
            detectors.append("cusum_warn")
        elif cusum_severity is DriftSeverity.alert:
            detectors.append("cusum_alert")
        if variance_severity is DriftSeverity.warn:
            detectors.append("variance_warn")
        elif variance_severity is DriftSeverity.alert:
            detectors.append("variance_alert")

        metric_payloads[name] = DriftSentryMetric(
            name=name,
            severity=metric_severity,
            baseline_mean=baseline_mean,
            evaluation_mean=evaluation_mean,
            baseline_variance=baseline_variance,
            evaluation_variance=evaluation_variance,
            baseline_count=len(baseline),
            evaluation_count=len(evaluation),
            page_hinkley_stat=page_stat,
            cusum_stat=cusum_stat,
            variance_ratio=variance_ratio,
            detectors=tuple(detectors),
        )
        status = _max_severity(status, metric_severity)
        severity_counts[metric_severity.value] += 1
        detector_entry: dict[str, object] = {"severity": metric_severity.value}
        if detectors:
            detector_entry["detectors"] = list(detectors)
        if page_stat is not None:
            detector_entry["page_hinkley_stat"] = page_stat
        if cusum_stat is not None:
            detector_entry["cusum_stat"] = cusum_stat
        if variance_ratio is not None:
            detector_entry["variance_ratio"] = variance_ratio
        detector_catalog[name] = detector_entry

    if not metric_payloads:
        raise ValueError("No metrics satisfied the drift sentry evaluation windows")

    snapshot_metadata: dict[str, object] = {
        "metric_count": len(metric_payloads),
        "runbook": "docs/operations/runbooks/drift_sentry_response.md",
    }
    if severity_counts:
        snapshot_metadata["severity_counts"] = dict(severity_counts)
    if detector_catalog:
        snapshot_metadata["detectors"] = detector_catalog
    if metadata:
        snapshot_metadata.update(dict(metadata))

    return DriftSentrySnapshot(
        generated_at=generated,
        status=status,
        metrics=metric_payloads,
        config=cfg,
        metadata=snapshot_metadata,
    )


def _should_emit(severity: DriftSeverity, threshold: DriftSeverity) -> bool:
    return _SEVERITY_ORDER[severity] >= _SEVERITY_ORDER[threshold]


def derive_drift_sentry_alerts(
    snapshot: DriftSentrySnapshot,
    *,
    threshold: DriftSeverity = DriftSeverity.warn,
    include_overall: bool = True,
    base_tags: Sequence[str] = ("drift-sentry",),
) -> list[AlertEvent]:
    """Translate drift sentry telemetry into alert events."""

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
                category="understanding.drift_sentry",
                severity=severity_map[snapshot.status],
                message=f"Drift sentry status {snapshot.status.value}",
                tags=base_tag_tuple,
                context={"snapshot": snapshot.as_dict()},
            )
        )

    for name, metric in snapshot.metrics.items():
        if not _should_emit(metric.severity, threshold):
            continue
        tags = base_tag_tuple + (name,)
        detector_suffix = f" ({', '.join(metric.detectors)})" if metric.detectors else ""
        events.append(
            AlertEvent(
                category="understanding.drift_sentry",
                severity=severity_map[metric.severity],
                message=f"{name} drift {metric.severity.value}{detector_suffix}",
                tags=tags,
                context={
                    "metric": metric.as_dict(),
                    "snapshot": snapshot.as_dict(),
                },
            )
        )

    return events


def publish_drift_sentry_snapshot(
    event_bus: EventBus,
    snapshot: DriftSentrySnapshot,
    *,
    source: str = "operations.drift_sentry",
    global_bus_factory: Callable[[], TopicBus] | None = None,
) -> None:
    """Publish the drift sentry snapshot to the runtime bus with failover."""

    event = Event(
        type="telemetry.understanding.drift_sentry",
        payload=snapshot.as_dict(),
        source=source,
    )

    publish_event_with_failover(
        event_bus,
        event,
        logger=logger,
        runtime_fallback_message="Runtime bus rejected drift sentry snapshot; using global bus",
        runtime_unexpected_message="Unexpected error publishing drift sentry snapshot via runtime bus",
        runtime_none_message="Runtime bus returned None publishing drift sentry snapshot; using global bus",
        global_not_running_message="Global event bus not running while publishing drift sentry snapshot",
        global_unexpected_message="Unexpected error publishing drift sentry snapshot via global bus",
        global_bus_factory=global_bus_factory,
    )


__all__ = [
    "DriftSentryConfig",
    "DriftSentryMetric",
    "DriftSentrySnapshot",
    "derive_drift_sentry_alerts",
    "evaluate_drift_sentry",
    "publish_drift_sentry_snapshot",
]
