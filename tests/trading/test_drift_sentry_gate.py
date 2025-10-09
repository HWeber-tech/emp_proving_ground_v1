from dataclasses import dataclass
from datetime import datetime, timezone

import datetime as _datetime
import enum as _enum
import typing as _typing

if not hasattr(_datetime, "UTC"):
    _datetime.UTC = _datetime.timezone.utc  # type: ignore[attr-defined]

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

import pytest

from src.operations.sensory_drift import DriftSeverity, SensoryDimensionDrift, SensoryDriftSnapshot
from src.trading.gating import DriftSentryGate


def _make_snapshot(severity: DriftSeverity) -> SensoryDriftSnapshot:
    dimension = SensoryDimensionDrift(
        name="WHY",
        current_signal=0.42,
        baseline_signal=0.10,
        delta=0.32,
        current_confidence=0.76,
        baseline_confidence=0.70,
        confidence_delta=0.06,
        severity=severity,
        samples=6,
    )
    return SensoryDriftSnapshot(
        generated_at=datetime(2024, 1, 5, tzinfo=timezone.utc),
        status=severity,
        dimensions={"WHY": dimension},
        sample_window=6,
        metadata={"source": "test"},
    )


def test_gate_allows_without_snapshot() -> None:
    gate = DriftSentryGate()
    decision = gate.evaluate_trade(
        symbol="EURUSD",
        strategy_id="alpha",
        confidence=0.9,
        quantity=1.0,
        notional=1_000.0,
        metadata={},
    )
    assert decision.allowed
    assert decision.reason == "no_snapshot"


def test_warn_blocks_low_confidence_trade() -> None:
    gate = DriftSentryGate(warn_confidence_floor=0.7)
    gate.update_snapshot(_make_snapshot(DriftSeverity.warn))

    decision = gate.evaluate_trade(
        symbol="EURUSD",
        strategy_id="alpha",
        confidence=0.5,
        quantity=1.0,
        notional=10_000.0,
        metadata={},
    )
    assert not decision.allowed
    assert "confidence" in (decision.reason or "")

    allowed = gate.evaluate_trade(
        symbol="EURUSD",
        strategy_id="alpha",
        confidence=0.9,
        quantity=1.0,
        notional=10_000.0,
        metadata={},
    )
    assert allowed.allowed


def test_alert_blocks_regardless_of_confidence() -> None:
    gate = DriftSentryGate()
    gate.update_snapshot(_make_snapshot(DriftSeverity.alert))

    decision = gate.evaluate_trade(
        symbol="EURUSD",
        strategy_id="alpha",
        confidence=0.95,
        quantity=1.0,
        notional=5_000.0,
        metadata={},
    )
    assert not decision.allowed
    assert decision.severity is DriftSeverity.alert
    assert "block" in (decision.reason or "")


def test_exempt_strategy_bypasses_gate() -> None:
    gate = DriftSentryGate(exempt_strategies=["alpha"])
    gate.update_snapshot(_make_snapshot(DriftSeverity.alert))

    decision = gate.evaluate_trade(
        symbol="EURUSD",
        strategy_id="alpha",
        confidence=0.2,
        quantity=1.0,
        notional=5_000.0,
        metadata={},
    )
    assert decision.allowed
    assert decision.reason == "strategy_exempt"


@pytest.mark.parametrize(
    "notional,expected",
    [
        (10_000.0, True),
        (100_000.0, False),
    ],
)
def test_warn_notional_limit(notional: float, expected: bool) -> None:
    gate = DriftSentryGate(warn_notional_limit=25_000.0)
    gate.update_snapshot(_make_snapshot(DriftSeverity.warn))

    decision = gate.evaluate_trade(
        symbol="EURUSD",
        strategy_id="alpha",
        confidence=0.9,
        quantity=1.0,
        notional=notional,
        metadata={},
    )
    assert decision.allowed is expected


def test_adaptive_threshold_overrides() -> None:
    gate = DriftSentryGate(warn_confidence_floor=0.6)
    gate.update_snapshot(_make_snapshot(DriftSeverity.warn))

    blocked = gate.evaluate_trade(
        symbol="EURUSD",
        strategy_id="alpha",
        confidence=0.7,
        quantity=1.0,
        notional=5_000.0,
        metadata={},
        threshold_overrides={
            "warn_confidence_floor": 0.8,
            "stage": "pilot",
        },
    )
    assert not blocked.allowed
    assert "0.800" in (blocked.reason or "")
    assert blocked.requirements["release_stage"] == "pilot"

    allowed = gate.evaluate_trade(
        symbol="EURUSD",
        strategy_id="alpha",
        confidence=0.9,
        quantity=1.0,
        notional=5_000.0,
        metadata={},
        threshold_overrides={
            "warn_confidence_floor": 0.65,
        },
    )
    assert allowed.allowed


def test_stage_gate_forces_paper_for_paper_stage() -> None:
    gate = DriftSentryGate()
    gate.update_snapshot(_make_snapshot(DriftSeverity.normal))

    decision = gate.evaluate_trade(
        symbol="EURUSD",
        strategy_id="shadow-alpha",
        confidence=0.92,
        quantity=1.0,
        notional=5_000.0,
        metadata={},
        threshold_overrides={"stage": "paper"},
    )

    assert decision.allowed is True
    assert decision.force_paper is True
    assert decision.reason == "release_stage_paper_requires_paper_execution"
    requirements = dict(decision.requirements)
    assert requirements["release_stage"] == "paper"
    assert (
        requirements["release_stage_gate"]
        == "release_stage_paper_requires_paper_execution"
    )
