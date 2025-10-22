from __future__ import annotations

import math
from collections import deque

import pytest

from src.evolution.optimization.preferences import (
    PreferenceArticulator,
    PreferenceProfile,
    interactive_preference_tuning,
)


@pytest.fixture()
def base_profile() -> PreferenceProfile:
    profile = PreferenceProfile()
    profile = profile.with_preference("sharpe", weight=3.0)
    profile = profile.with_preference("drawdown", weight=1.0)
    profile = profile.with_preference("turnover", weight=0.5)
    return profile


def test_normalized_weights_respect_relative_importance(base_profile: PreferenceProfile) -> None:
    weights = base_profile.normalized_weights()
    assert pytest.approx(weights["sharpe"], rel=1e-6) == 3.0 / 4.5
    assert pytest.approx(weights["drawdown"], rel=1e-6) == 1.0 / 4.5
    assert pytest.approx(weights["turnover"], rel=1e-6) == 0.5 / 4.5

    metrics = {"sharpe": 1.0, "drawdown": -0.2, "turnover": -0.1}
    score = base_profile.score(metrics)
    expected = weights["sharpe"] * 1.0 + weights["drawdown"] * -0.2 + weights["turnover"] * -0.1
    assert pytest.approx(score, rel=1e-6) == expected


def test_score_handles_missing_metrics(base_profile: PreferenceProfile) -> None:
    metrics = {"sharpe": 0.5}
    score = base_profile.score(metrics, missing_value=-0.5)
    weights = base_profile.normalized_weights()
    expected = weights["sharpe"] * 0.5 + weights["drawdown"] * -0.5 + weights["turnover"] * -0.5
    assert pytest.approx(score, rel=1e-6) == expected


def test_interactive_tuning_updates_weights(base_profile: PreferenceProfile) -> None:
    responses = deque(["5", "", "1.5"])

    def fake_input(prompt: str) -> str:  # pragma: no cover - executed in tests
        return responses.popleft()

    captured: list[str] = []

    def fake_output(message: str) -> None:  # pragma: no cover - executed in tests
        captured.append(message)

    tuned = interactive_preference_tuning(
        base_profile,
        input_fn=fake_input,
        output_fn=fake_output,
    )

    assert pytest.approx(tuned.get("sharpe").weight, rel=1e-6) == 5.0
    assert pytest.approx(tuned.get("drawdown").weight, rel=1e-6) == base_profile.get("drawdown").weight
    assert pytest.approx(tuned.get("turnover").weight, rel=1e-6) == 1.5
    assert any("Updated preferences" in message for message in captured)


def test_articulator_respects_preference_shifts(base_profile: PreferenceProfile) -> None:
    articulator = PreferenceArticulator(base_profile)
    metrics = {"sharpe": 1.0, "drawdown": -1.0, "turnover": -0.1}

    original_score = articulator.articulate(metrics)
    articulator.update("drawdown", weight=5.0)
    adjusted_score = articulator.articulate(metrics)

    assert adjusted_score < original_score
    articulator.update("sharpe", weight=0.5)
    final_score = articulator.articulate(metrics)
    assert final_score < adjusted_score
    assert math.isfinite(final_score)
