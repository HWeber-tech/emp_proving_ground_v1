from __future__ import annotations

import pytest

from src.evolution.feature_flags import ADAPTIVE_RUNS_FLAG, EvolutionFeatureFlags


@pytest.fixture(autouse=True)
def _clear_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ADAPTIVE_RUNS_FLAG, raising=False)


def test_adaptive_runs_disabled_by_default() -> None:
    flags = EvolutionFeatureFlags()
    assert flags.adaptive_runs_enabled() is False


def test_adaptive_runs_enabled_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ADAPTIVE_RUNS_FLAG, "yes")
    flags = EvolutionFeatureFlags()
    assert flags.adaptive_runs_enabled() is True


def test_adaptive_runs_override_and_custom_env() -> None:
    flags = EvolutionFeatureFlags(env={ADAPTIVE_RUNS_FLAG: "on"})
    assert flags.adaptive_runs_enabled() is True
    assert flags.adaptive_runs_enabled(override=False) is False
