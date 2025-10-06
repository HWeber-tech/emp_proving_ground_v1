import datetime as _datetime
import enum as _enum
import typing as _typing

if not hasattr(_enum, "StrEnum"):
    class _StrEnum(str, _enum.Enum):
        pass

    _enum.StrEnum = _StrEnum


def _shim_class_getitem(name: str) -> type:
    class _Placeholder:
        @classmethod
        def __class_getitem__(cls, item):
            return item

    _Placeholder.__name__ = name
    return _Placeholder


if not hasattr(_typing, "Unpack"):
    _typing.Unpack = _shim_class_getitem("Unpack")  # type: ignore[attr-defined]

if not hasattr(_typing, "NotRequired"):
    _typing.NotRequired = _shim_class_getitem("NotRequired")  # type: ignore[attr-defined]

if not hasattr(_datetime, "UTC"):
    _datetime.UTC = _datetime.timezone.utc  # type: ignore[attr-defined]

from datetime import UTC, datetime

import pytest

from src.operations.drift_sentry import (
    DriftSentryConfig,
    evaluate_drift_sentry,
    derive_drift_sentry_alerts,
)
from src.operations.sensory_drift import DriftSeverity


def test_drift_sentry_detects_abrupt_shift() -> None:
    config = DriftSentryConfig(
        baseline_window=12,
        evaluation_window=4,
        min_observations=4,
        page_hinkley_delta=0.001,
        page_hinkley_warn=0.2,
        page_hinkley_alert=0.4,
        variance_ratio_warn=1.2,
        variance_ratio_alert=1.6,
    )
    series = [0.05] * 12 + [0.9] * 4

    snapshot = evaluate_drift_sentry(
        {"belief_confidence": series},
        config=config,
        generated_at=datetime(2025, 1, 1, tzinfo=UTC),
    )

    metric = snapshot.metrics["belief_confidence"]
    assert snapshot.status is DriftSeverity.alert
    assert metric.severity is DriftSeverity.alert
    assert "page_hinkley_alert" in metric.detectors
    assert metric.variance_ratio is not None


def test_drift_sentry_variance_warn() -> None:
    config = DriftSentryConfig(
        baseline_window=8,
        evaluation_window=4,
        min_observations=4,
        page_hinkley_delta=0.01,
        page_hinkley_warn=0.8,
        page_hinkley_alert=1.8,
        variance_ratio_warn=1.6,
        variance_ratio_alert=5.5,
    )

    baseline = [
        1.0,
        1.02,
        0.98,
        1.01,
        0.99,
        1.03,
        0.97,
        1.02,
    ]
    evaluation = [0.95, 1.05, 0.97, 1.03]

    snapshot = evaluate_drift_sentry(
        {"regime_variance": baseline + evaluation},
        config=config,
        generated_at=datetime(2025, 1, 2, tzinfo=UTC),
    )

    metric = snapshot.metrics["regime_variance"]
    assert metric.severity is DriftSeverity.warn
    assert "variance_warn" in metric.detectors
    assert metric.page_hinkley_stat is not None


def test_drift_sentry_alert_generation_respects_threshold() -> None:
    config = DriftSentryConfig(
        baseline_window=6,
        evaluation_window=3,
        min_observations=3,
        page_hinkley_delta=0.005,
        page_hinkley_warn=0.18,
        page_hinkley_alert=0.5,
        variance_ratio_warn=1.15,
        variance_ratio_alert=2.5,
    )

    baseline = [0.19, 0.21, 0.2, 0.18, 0.22, 0.2]
    evaluation = [0.26, 0.3, 0.28]
    series = baseline + evaluation
    snapshot = evaluate_drift_sentry(
        {"belief_bias": series},
        config=config,
        generated_at=datetime(2025, 1, 3, tzinfo=UTC),
    )

    events = derive_drift_sentry_alerts(snapshot)
    categories = {event.category for event in events}
    assert categories == {"understanding.drift_sentry"}
    assert any("belief_bias" in event.tags for event in events)

    suppressed = derive_drift_sentry_alerts(
        snapshot,
        threshold=DriftSeverity.alert,
    )
    assert suppressed == []


def test_drift_sentry_requires_sufficient_observations() -> None:
    config = DriftSentryConfig(
        baseline_window=5,
        evaluation_window=3,
        min_observations=3,
    )

    with pytest.raises(ValueError):
        evaluate_drift_sentry({"belief": [0.1, 0.2, 0.3]}, config=config)
