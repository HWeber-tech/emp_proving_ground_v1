from __future__ import annotations

import pytest

from src.thinking.adaptation.fast_weights import (
    FastWeightConstraints,
    FastWeightController,
    build_fast_weight_controller,
    parse_fast_weight_constraints,
)


def test_controller_clamps_inhibitory_when_disabled() -> None:
    controller = FastWeightController(
        FastWeightConstraints(
            allow_inhibitory=False,
            activation_threshold=1.05,
        )
    )

    result = controller.constrain(
        fast_weights={"alpha": 0.4, "beta": 1.3},
        tactic_ids=("alpha", "beta"),
    )

    weights = result.weights
    # `alpha` multiplier is suppressed back to baseline so it is omitted from the
    # constrained payload, while the excitatory `beta` multiplier survives.
    assert "alpha" not in weights
    assert weights["beta"] == pytest.approx(1.3)

    metrics = result.metrics.as_dict()
    assert metrics["inhibitory"] == 0
    assert metrics["suppressed_inhibitory"] == 1
    assert metrics["suppressed_inhibitory_ids"] == ("alpha",)
    assert metrics["min_multiplier"] == pytest.approx(1.0)


def test_controller_tracks_inhibitory_when_allowed() -> None:
    controller = FastWeightController(
        FastWeightConstraints(
            allow_inhibitory=True,
            activation_threshold=1.1,
            prune_tolerance=1e-6,
        )
    )

    assert result.weights == {}
    assert result.metrics.total == 2
    assert result.metrics.active == 0
    assert result.metrics.active_percentage == pytest.approx(0.0)
    assert result.metrics.sparsity == pytest.approx(1.0)
    assert result.metrics.active_ids == ()
    assert result.metrics.dormant_ids == ("alpha", "bravo")
    assert result.metrics.max_multiplier == pytest.approx(1.0)
    assert result.metrics.min_multiplier == pytest.approx(1.0)


def test_constraints_parser_builds_overrides_from_mapping() -> None:
    overrides = {
        "fast_weight_baseline": "1.2",
        "FAST_WEIGHT_MINIMUM_MULTIPLIER": "0.5",
        "FAST_WEIGHT_ACTIVATION_THRESHOLD": "1.3",
        "FAST_WEIGHT_MAX_ACTIVE_FRACTION": "0.25",
        "FAST_WEIGHT_PRUNE_TOLERANCE": "0.0001",
        "FAST_WEIGHT_EXCITATORY_ONLY": "true",
    }

    constraints = parse_fast_weight_constraints(overrides)
    assert constraints is not None
    assert constraints.baseline == pytest.approx(1.2)
    assert constraints.minimum_multiplier == pytest.approx(0.5)
    assert constraints.activation_threshold == pytest.approx(1.3)
    assert constraints.max_active_fraction == pytest.approx(0.25)
    assert constraints.prune_tolerance == pytest.approx(0.0001)
    assert constraints.excitatory_only is True


def test_controller_excitatory_mode_clamps_below_baseline() -> None:
    controller = build_fast_weight_controller(
        {
            "FAST_WEIGHT_EXCITATORY_ONLY": "1",
            "FAST_WEIGHT_ACTIVATION_THRESHOLD": "1.1",
        }
    )
    result = controller.constrain(
        fast_weights={"alpha": 0.25, "bravo": -1.0},
        tactic_ids=("alpha", "bravo", "charlie"),
    )

    assert result.weights == {}
    metrics = result.metrics
    assert metrics.min_multiplier == pytest.approx(1.0)
    assert metrics.max_multiplier == pytest.approx(1.0)
    assert metrics.active == 0
