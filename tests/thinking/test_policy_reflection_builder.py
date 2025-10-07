from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.thinking.adaptation.policy_reflection import (
    PolicyReflectionBuilder,
)
from src.thinking.adaptation.policy_router import (
    FastWeightExperiment,
    PolicyRouter,
    PolicyTactic,
    RegimeState,
)


UTC = timezone.utc


def _regime(ts: datetime) -> RegimeState:
    return RegimeState(
        regime="bull",
        confidence=0.8,
        features={"volume_z": 0.1, "volatility": 0.2},
        timestamp=ts,
    )


def test_builder_handles_empty_history() -> None:
    router = PolicyRouter()
    builder = PolicyReflectionBuilder(router, now=lambda: datetime(2024, 1, 1, tzinfo=UTC))

    artifacts = builder.build()

    assert artifacts.digest["total_decisions"] == 0
    assert "No decisions" in artifacts.markdown
    assert artifacts.payload["metadata"]["total_decisions"] == 0


def test_builder_generates_markdown_with_insights() -> None:
    base_ts = datetime(2024, 3, 15, 12, 0, tzinfo=UTC)
    router = PolicyRouter()
    router.register_tactics(
        (
            PolicyTactic(
                tactic_id="breakout",
                base_weight=1.0,
                regime_bias={"bull": 1.3},
                description="Momentum breakout",
                objectives=("alpha",),
                tags=("momentum",),
            ),
            PolicyTactic(
                tactic_id="mean_reversion",
                base_weight=0.9,
                regime_bias={"bull": 1.0},
                description="Reversion",
                objectives=("stability",),
                tags=("reversion",),
            ),
        )
    )
    router.register_experiment(
        FastWeightExperiment(
            experiment_id="exp-boost",
            tactic_id="mean_reversion",
            delta=0.5,
            rationale="Promote reversion in calm regimes",
            feature_gates={"volatility": (None, 0.3)},
            min_confidence=0.6,
            regimes=("bull",),
        )
    )

    router.route(_regime(base_ts))
    router.route(_regime(base_ts + timedelta(minutes=3)))
    router.route(_regime(base_ts + timedelta(minutes=6)))

    builder = PolicyReflectionBuilder(
        router,
        now=lambda: datetime(2024, 3, 15, 12, 15, tzinfo=UTC),
        default_window=5,
    )
    artifacts = builder.build()

    markdown = artifacts.markdown
    assert "PolicyRouter reflection summary" in markdown
    assert "Decisions analysed: 3" in markdown
    assert "Top tactics" in markdown
    assert "Active experiments" in markdown
    assert "Tag spotlight" in markdown
    assert "Objective coverage" in markdown
    assert "exp-boost" in markdown
    assert "Conf >=" in markdown

    insights = artifacts.payload["insights"]
    assert any("Leading tactic" in insight for insight in insights)
    assert any("Top experiment exp-boost" in insight for insight in insights)
    assert any("Dominant tag" in insight for insight in insights)
    assert any("Leading objective" in insight for insight in insights)
    assert any("confidence >= 0.60" in insight for insight in insights)

    digest = artifacts.payload["digest"]
    experiments = digest["experiments"]
    assert experiments[0]["regimes"] == ["bull"]
    assert experiments[0]["min_confidence"] == pytest.approx(0.6)
