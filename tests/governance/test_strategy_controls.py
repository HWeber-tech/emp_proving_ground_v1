import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.governance.strategy_controls import (
    StrategyControlError,
    StrategyControls,
)


@pytest.fixture()
def controls(tmp_path: Path) -> StrategyControls:
    path = tmp_path / "strategy_controls.json"
    return StrategyControls(path, auto_flush=True)


def test_pause_and_resume_controls(controls: StrategyControls) -> None:
    controls.pause_strategy("momentum_v1", reason="maintenance", actor="ops")
    state = controls.get_state("momentum_v1")
    assert state is not None
    assert state.paused is True
    assert state.pause_reason == "maintenance"
    assert controls.is_paused("momentum_v1") is True

    controls.resume_strategy("momentum_v1", actor="ops")
    resumed = controls.get_state("momentum_v1")
    assert resumed is not None
    assert resumed.paused is False
    assert resumed.pause_reason is None
    assert controls.is_paused("momentum_v1") is False


def test_resume_rejected_when_quarantined(controls: StrategyControls) -> None:
    controls.quarantine_strategy("strategy_a", reason="risk breach", actor="ops")
    assert controls.is_quarantined("strategy_a") is True

    with pytest.raises(StrategyControlError):
        controls.resume_strategy("strategy_a")

    controls.release_quarantine("strategy_a", actor="ops")
    assert controls.is_quarantined("strategy_a") is False
    controls.resume_strategy("strategy_a")
    assert controls.is_paused("strategy_a") is False


def test_quarantine_expiry_normalises(controls: StrategyControls) -> None:
    expiry = datetime.now(tz=UTC) + timedelta(hours=1)
    controls.quarantine_strategy("strategy_b", reason="ops", expires_at=expiry.isoformat())
    state = controls.get_state("strategy_b")
    assert state is not None
    assert state.quarantine_expires_at is not None
    assert state.quarantine_expires_at.tzinfo is not None


def test_risk_limit_breach_detection(controls: StrategyControls) -> None:
    controls.set_risk_limits(
        "strategy_risk",
        {
            "max_drawdown": 0.12,
            "max_leverage": 2.0,
        },
    )

    ok, breaches = controls.check_risk_limits(
        "strategy_risk",
        {"drawdown": 0.1, "leverage": 1.5},
    )
    assert ok is True
    assert breaches == ()

    ok, breaches = controls.check_risk_limits(
        "strategy_risk",
        {"drawdown": 0.15, "leverage": 1.5},
    )
    assert ok is False
    assert len(breaches) == 1
    assert breaches[0].limit == "max_drawdown"
    assert breaches[0].threshold == 0.12

    with pytest.raises(StrategyControlError):
        controls.enforce_risk_limits("strategy_risk", {"drawdown": 0.2})


def test_persistence_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "controls.json"
    controls = StrategyControls(path, auto_flush=True)
    controls.pause_strategy("strategy_c", reason="ops")
    controls.set_risk_limits("strategy_c", {"max_notional": 1000000})
    controls.quarantine_strategy("strategy_c", reason="investigation")

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "strategies" in raw
    assert "strategy_c" in raw["strategies"]

    reloaded = StrategyControls(path, auto_flush=False)
    state = reloaded.get_state("strategy_c")
    assert state is not None
    assert state.paused is True
    assert state.risk_limits["max_notional"] == 1000000
    assert state.quarantined is True


def test_invalid_risk_limit_rejected(controls: StrategyControls) -> None:
    with pytest.raises(StrategyControlError):
        controls.set_risk_limits("bad_strategy", {"max_drawdown": "not-a-number"})

    with pytest.raises(StrategyControlError):
        controls.set_risk_limits("bad_strategy", {"max_drawdown": float("inf")})

    with pytest.raises(StrategyControlError):
        controls.set_risk_limits("bad_strategy", {"max_drawdown": -1.0})
