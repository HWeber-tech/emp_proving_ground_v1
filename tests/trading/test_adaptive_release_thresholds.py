from datetime import datetime, timezone
from pathlib import Path

import typing as _typing

if not hasattr(_typing, "Unpack"):
    class _Placeholder:
        @classmethod
        def __class_getitem__(cls, item):
            return item

    _typing.Unpack = _Placeholder  # type: ignore[attr-defined,assignment]

if not hasattr(_typing, "NotRequired"):
    class _PlaceholderOptional:
        @classmethod
        def __class_getitem__(cls, item):
            return item

    _typing.NotRequired = _PlaceholderOptional  # type: ignore[attr-defined,assignment]

import pytest

from src.governance.policy_ledger import (
    LedgerReleaseManager,
    PolicyLedgerStage,
    PolicyLedgerStore,
)
from src.operations.sensory_drift import DriftSeverity, SensoryDimensionDrift, SensoryDriftSnapshot
from src.trading.gating.adaptive_release import AdaptiveReleaseThresholds


def _snapshot(
    status: DriftSeverity,
    dimensions: dict[str, SensoryDimensionDrift],
    metadata: dict[str, object] | None = None,
) -> SensoryDriftSnapshot:
    base_metadata: dict[str, object] = {"source": "test"}
    if metadata:
        base_metadata.update(metadata)
    return SensoryDriftSnapshot(
        generated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        status=status,
        dimensions=dimensions,
        sample_window=60,
        metadata=base_metadata,
    )


def _dimension(name: str, severity: DriftSeverity) -> SensoryDimensionDrift:
    return SensoryDimensionDrift(
        name=name,
        current_signal=0.4,
        baseline_signal=0.1,
        delta=0.3,
        current_confidence=0.7,
        baseline_confidence=0.6,
        confidence_delta=0.1,
        severity=severity,
        samples=12,
    )


def test_adaptive_thresholds_tighten_on_warn(tmp_path: Path) -> None:
    store = PolicyLedgerStore(tmp_path / "ledger.json")
    manager = LedgerReleaseManager(store)
    manager.promote(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.PAPER,
        threshold_overrides={
            "warn_confidence_floor": 0.8,
            "warn_notional_limit": 50_000.0,
        },
        evidence_id="diary-alpha",
    )

    resolver = AdaptiveReleaseThresholds(manager)
    snapshot = _snapshot(
        DriftSeverity.warn,
        {
            "HOW": _dimension("HOW", DriftSeverity.warn),
        },
    )

    thresholds = resolver.resolve(strategy_id="alpha", snapshot=snapshot)
    assert thresholds["stage"] == PolicyLedgerStage.PAPER.value
    assert thresholds["warn_confidence_floor"] > 0.8
    assert thresholds["warn_notional_limit"] < 50_000.0
    assert thresholds["block_severity"] == DriftSeverity.alert.value
    assert thresholds["adaptive_source"] == DriftSeverity.warn.value


def test_adaptive_thresholds_escalate_on_alert(tmp_path: Path) -> None:
    store = PolicyLedgerStore(tmp_path / "ledger.json")
    manager = LedgerReleaseManager(store)
    manager.promote(
        policy_id="beta",
        tactic_id="beta",
        stage=PolicyLedgerStage.EXPERIMENT,
        threshold_overrides={
            "warn_confidence_floor": 0.85,
            "warn_notional_limit": 20_000.0,
            "block_severity": "warn",
        },
        evidence_id="diary-beta",
    )

    resolver = AdaptiveReleaseThresholds(manager)
    snapshot = _snapshot(
        DriftSeverity.alert,
        {
            "WHAT": _dimension("WHAT", DriftSeverity.alert),
            "WHY": _dimension("WHY", DriftSeverity.warn),
        },
    )

    thresholds = resolver.resolve(strategy_id="beta", snapshot=snapshot)
    assert thresholds["block_severity"] == DriftSeverity.warn.value
    assert thresholds["warn_confidence_floor"] >= 0.9
    assert thresholds["warn_notional_limit"] <= 10_000.0
    assert thresholds["adaptive_source"] == DriftSeverity.alert.value


def test_adaptive_thresholds_without_snapshot(tmp_path: Path) -> None:
    store = PolicyLedgerStore(tmp_path / "ledger.json")
    manager = LedgerReleaseManager(store)
    manager.promote(
        policy_id="gamma",
        tactic_id="gamma",
        stage=PolicyLedgerStage.PILOT,
        threshold_overrides={
            "warn_confidence_floor": "0.72",
            "warn_notional_limit": 75_000,
        },
        evidence_id="diary-gamma",
    )

    resolver = AdaptiveReleaseThresholds(manager)
    thresholds = resolver.resolve(strategy_id="gamma", snapshot=None)
    assert thresholds["stage"] == PolicyLedgerStage.PAPER.value
    assert thresholds["warn_confidence_floor"] == pytest.approx(0.85, rel=1e-6)
    assert thresholds["warn_notional_limit"] == pytest.approx(50_000.0)
    assert thresholds["block_severity"] == DriftSeverity.alert.value
    assert thresholds["adaptive_source"] == "sensor_unavailable"
    assert thresholds["uncertainty_inflation"] == pytest.approx(0.1, rel=1e-6)


def test_adaptive_thresholds_with_failure_metadata(tmp_path: Path) -> None:
    store = PolicyLedgerStore(tmp_path / "ledger.json")
    manager = LedgerReleaseManager(store)
    manager.promote(
        policy_id="delta",
        tactic_id="delta",
        stage=PolicyLedgerStage.PAPER,
        threshold_overrides={
            "warn_confidence_floor": 0.75,
            "warn_notional_limit": 40_000.0,
        },
        evidence_id="diary-delta",
    )

    resolver = AdaptiveReleaseThresholds(manager)
    snapshot = _snapshot(
        DriftSeverity.normal,
        {},
        metadata={"reason": "no_audit_entries", "samples": 0},
    )

    thresholds = resolver.resolve(strategy_id="delta", snapshot=snapshot)
    assert thresholds["warn_confidence_floor"] == pytest.approx(0.85, rel=1e-6)
    assert thresholds["adaptive_source"] == "no_audit_entries"
    assert thresholds["uncertainty_inflation"] == pytest.approx(0.1, rel=1e-6)


def test_sensor_failure_reverts_to_baseline_policy(tmp_path: Path) -> None:
    store = PolicyLedgerStore(tmp_path / "ledger.json")
    manager = LedgerReleaseManager(store)
    manager.promote(
        policy_id="omega",
        tactic_id="omega",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        threshold_overrides={
            "warn_confidence_floor": 0.55,
            "warn_notional_limit": 150_000.0,
        },
        approvals=("risk", "ops"),
        evidence_id="diary-omega",
        metadata={
            "promotion_checklist": {
                "oos_regime_grid": True,
                "leakage_checks": True,
                "risk_audit": True,
            }
        },
    )

    resolver = AdaptiveReleaseThresholds(manager)
    thresholds = resolver.resolve(strategy_id="omega", snapshot=None)

    assert thresholds["stage"] == PolicyLedgerStage.PAPER.value
    assert thresholds.get("reverted_from_stage") == PolicyLedgerStage.LIMITED_LIVE.value
    assert thresholds["adaptive_source"] == "sensor_unavailable"
    assert thresholds["warn_confidence_floor"] == pytest.approx(0.85, rel=1e-6)
    assert thresholds["warn_notional_limit"] == pytest.approx(50_000.0)
    assert thresholds["block_severity"] == DriftSeverity.alert.value
    assert thresholds["uncertainty_inflation"] == pytest.approx(0.1, rel=1e-6)
