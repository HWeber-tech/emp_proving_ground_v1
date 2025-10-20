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
        cusum_drift=0.0,
        cusum_warn=1.5,
        cusum_alert=2.5,
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
    assert "cusum_alert" in metric.detectors
    assert metric.variance_ratio is not None
    assert metric.cusum_stat is not None

    metadata = snapshot.metadata
    actions = metadata.get("actions")
    assert isinstance(actions, list) and actions
    action_labels = {entry.get("action") for entry in actions}
    assert {"freeze_exploration", "size_multiplier"}.issubset(action_labels)
    size_action = next(entry for entry in actions if entry.get("action") == "size_multiplier")
    assert pytest.approx(size_action.get("value"), rel=1e-9) == 0.5

    snapshot_actions = snapshot.actions
    assert snapshot_actions
    snapshot_labels = {entry.get("action") for entry in snapshot_actions}
    assert snapshot_labels == action_labels

    theory_packet = metadata.get("theory_packet")
    assert isinstance(theory_packet, dict)
    assert theory_packet.get("severity") == "alert"
    assert theory_packet.get("actions") == actions
    triggers = theory_packet.get("triggers")
    assert isinstance(triggers, list) and triggers
    assert triggers[0].get("metric") == "belief_confidence"


def test_drift_sentry_variance_warn() -> None:
    config = DriftSentryConfig(
        baseline_window=8,
        evaluation_window=4,
        min_observations=4,
        page_hinkley_delta=0.01,
        page_hinkley_warn=0.8,
        page_hinkley_alert=1.8,
        cusum_drift=0.0,
        cusum_warn=5.0,
        cusum_alert=7.5,
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

    metadata = snapshot.metadata
    theory_packet = metadata.get("theory_packet")
    assert isinstance(theory_packet, dict)
    assert theory_packet.get("severity") == "warn"
    actions = metadata.get("actions")
    assert isinstance(actions, list)
    assert any(action.get("action") == "freeze_exploration" for action in actions)
    assert snapshot.actions


def test_drift_sentry_alert_generation_respects_threshold() -> None:
    config = DriftSentryConfig(
        baseline_window=6,
        evaluation_window=3,
        min_observations=3,
        page_hinkley_delta=0.005,
        page_hinkley_warn=0.18,
        page_hinkley_alert=0.5,
        cusum_warn=1.2,
        cusum_alert=2.0,
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

    metadata = snapshot.metadata
    assert metadata.get("actions"), "expected drift sentry actions metadata"
    packet = metadata.get("theory_packet")
    assert isinstance(packet, dict)
    assert packet.get("severity") == snapshot.status.value

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


def test_drift_sentry_training_divergence_adjusts_multiplier() -> None:
    config = DriftSentryConfig(
        baseline_window=8,
        evaluation_window=4,
        min_observations=4,
        page_hinkley_delta=10.0,
        page_hinkley_warn=1e6,
        page_hinkley_alert=1e6,
        cusum_warn=1e6,
        cusum_alert=1e6,
        variance_ratio_warn=float("inf"),
        variance_ratio_alert=float("inf"),
        training_mean_diff_warn=0.04,
        training_mean_diff_alert=0.1,
        training_variance_ratio_warn=float("inf"),
        training_variance_ratio_alert=float("inf"),
    )

    baseline = [0.1, 0.102, 0.099, 0.101, 0.098, 0.103, 0.097, 0.102]
    evaluation = [0.15, 0.16, 0.14, 0.15]
    series = baseline + evaluation
    training_reference = {"belief_confidence": [0.098, 0.101, 0.099, 0.102] * 6}

    snapshot = evaluate_drift_sentry(
        {"belief_confidence": series},
        config=config,
        generated_at=datetime(2025, 1, 6, tzinfo=UTC),
        training_reference=training_reference,
    )

    assert snapshot.status is DriftSeverity.warn

    metric = snapshot.metrics["belief_confidence"]
    assert metric.training_mean is not None
    assert metric.training_mean_delta is not None
    assert "training_mean_warn" in metric.detectors

    metadata = snapshot.metadata
    assert metadata.get("recommended_size_multiplier") == pytest.approx(0.35)

    training_block = metadata.get("training_divergence")
    assert isinstance(training_block, dict)
    assert training_block.get("status") == "warn"
    assert training_block.get("recommended_size_multiplier") == pytest.approx(0.35)

    actions = metadata.get("actions")
    assert isinstance(actions, list)
    size_action = next(entry for entry in actions if entry.get("action") == "size_multiplier")
    assert size_action.get("value") == pytest.approx(0.35)
    assert size_action.get("context_mult") == pytest.approx(0.35)

    assert snapshot.actions
    size_snapshot = next(entry for entry in snapshot.actions if entry.get("action") == "size_multiplier")
    assert size_snapshot.get("value") == pytest.approx(0.35)
