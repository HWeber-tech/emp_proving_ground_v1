from __future__ import annotations

import pytest

from src.evolution.feature_flags import (
    ADAPTIVE_RUNS_FLAG,
    AdaptiveRunDecision,
    EvolutionFeatureFlags,
)


@pytest.fixture(autouse=True)
def _clear_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ADAPTIVE_RUNS_FLAG, raising=False)


def test_adaptive_runs_disabled_by_default() -> None:
    flags = EvolutionFeatureFlags()
    decision = flags.adaptive_runs_decision()
    assert isinstance(decision, AdaptiveRunDecision)
    assert decision.enabled is False
    assert decision.source == "environment"
    assert decision.reason == "flag_missing"
    assert flags.adaptive_runs_enabled() is False


def test_adaptive_runs_enabled_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ADAPTIVE_RUNS_FLAG, "yes")
    flags = EvolutionFeatureFlags()
    decision = flags.adaptive_runs_decision()
    assert decision.enabled is True
    assert decision.source == "environment"
    assert decision.reason == "flag_enabled"
    assert decision.as_dict()["raw_value"] == "yes"
    assert flags.adaptive_runs_enabled() is True


def test_adaptive_runs_override_and_custom_env() -> None:
    flags = EvolutionFeatureFlags(env={ADAPTIVE_RUNS_FLAG: "on"})
    assert flags.adaptive_runs_enabled() is True

    override_decision = flags.adaptive_runs_decision(override=False)
    assert override_decision.enabled is False
    assert override_decision.source == "override"
    assert override_decision.reason == "override_disabled"
    assert override_decision.as_dict()["raw_value"] is False
