from __future__ import annotations

import pytest

from src.thinking.evaluation.retention_gates import (
    HorizonRetentionGate,
    RetentionGateDecision,
    evaluate_retention_gates,
)


def _caps() -> dict[str, float]:
    return {"ev1": 3.0, "EV5": 4.0, 20: 5.0}


def test_retention_gates_pass_when_within_caps() -> None:
    decision = evaluate_retention_gates({"EV1": 2.5, "5": 3.9, "ev20": 4.5}, _caps())

    assert isinstance(decision, RetentionGateDecision)
    assert decision.passed is True
    assert decision.missing_horizons == ()
    assert decision.extra_horizons == ()
    statuses = {gate.horizon: gate.status for gate in decision.gates}
    assert statuses == {"ev1": "pass", "ev5": "pass", "ev20": "pass"}


def test_retention_gates_fail_when_cap_exceeded() -> None:
    decision = evaluate_retention_gates({"ev1": 3.2, "ev5": 2.0, "ev20": 4.0}, _caps())

    assert decision.passed is False
    failing_gate = next(gate for gate in decision.gates if gate.horizon == "ev1")
    assert isinstance(failing_gate, HorizonRetentionGate)
    assert failing_gate.status == "fail"
    assert pytest.approx(failing_gate.exceeded_pct or 0.0, abs=1e-9) == pytest.approx(0.2)


def test_retention_gates_fail_when_missing_horizon() -> None:
    decision = evaluate_retention_gates({"ev1": 2.0}, {"ev1": 3.0, "ev5": 4.0})

    assert decision.passed is False
    assert decision.missing_horizons == ("ev5",)
    missing_gate = next(gate for gate in decision.gates if gate.horizon == "ev5")
    assert missing_gate.status == "missing"
    assert missing_gate.observed_pct is None


def test_retention_gates_record_extra_horizons() -> None:
    decision = evaluate_retention_gates(
        {"ev1": 2.0, "ev5": 3.0, "ev20": 4.0, "ev40": 1.0},
        {"ev1": 3.0, "ev5": 4.0, "ev20": 5.0},
    )

    assert decision.passed is True
    assert decision.extra_horizons == ("ev40",)


def test_retention_gates_support_negative_drop() -> None:
    decision = evaluate_retention_gates({"ev1": -1.0, "ev5": 0.0, "ev20": 4.0}, _caps())
    assert decision.passed is True


def test_retention_gates_fail_when_missing_all_observations() -> None:
    decision = evaluate_retention_gates({}, {"ev1": 3.0})
    assert decision.passed is False
    assert decision.missing_horizons == ("ev1",)


@pytest.mark.parametrize(
    "observed, caps",
    [
        ({"ev1": float("nan")}, {"ev1": 3.0}),
        ({"ev1": 2.0}, {}),
        ({"ev1": 2.0}, {"ev1": -1.0}),
        ({"ev1": 2.0}, {"ev1": 3.0, "EV1": 3.1}),
        ({"ev1": 2.0, "EV1": 1.0}, {"ev1": 3.0}),
    ],
)
def test_retention_gates_validate_invalid_inputs(observed, caps) -> None:
    with pytest.raises(ValueError):
        evaluate_retention_gates(observed, caps)


def test_retention_gate_decision_serialises() -> None:
    decision = evaluate_retention_gates({"ev1": 2.5, "ev5": 3.9, "ev20": 4.5}, _caps())

    payload = decision.as_dict()
    assert payload["passed"] is True
    assert payload["missing_horizons"] == []
    assert payload["extra_horizons"] == []
    assert len(payload["gates"]) == 3
    assert payload["gates"][0]["horizon"] == "ev1"
