from __future__ import annotations

import pytest

from src.thinking.adaptation.fast_weights import (
    FastWeightConstraints,
    FastWeightController,
    build_fast_weight_controller,
    parse_fast_weight_constraints,
)


def test_controller_clamps_negative_and_prunes_to_sparse_subset() -> None:
    controller = FastWeightController(
        FastWeightConstraints(activation_threshold=1.05, max_active_fraction=0.4)
    )
    tactics = ("alpha", "bravo", "charlie", "delta", "echo")
    result = controller.constrain(
        fast_weights={
            "alpha": 1.8,
            "bravo": 1.2,
            "charlie": 1.1,
            "delta": 0.5,
            "echo": -0.4,
        },
        tactic_ids=tactics,
    )

    assert set(result.weights) == {"alpha", "bravo", "delta", "echo"}
    assert result.weights["alpha"] == pytest.approx(1.8)
    assert result.weights["bravo"] == pytest.approx(1.2)
    assert result.weights["delta"] == pytest.approx(0.5)
    assert result.weights["echo"] == pytest.approx(0.0)
    assert all(value >= 0.0 for value in result.weights.values())

    metrics = result.metrics
    assert metrics.total == 5
    assert metrics.active == 2
    assert metrics.dormant == 3
    assert metrics.active_percentage == pytest.approx(40.0)
    assert metrics.sparsity == pytest.approx(0.6)
    assert metrics.active_ids == ("alpha", "bravo")
    assert metrics.dormant_ids == ("charlie", "delta", "echo")
    assert metrics.max_multiplier == pytest.approx(1.8)
    assert metrics.min_multiplier == pytest.approx(0.0)


def test_controller_returns_zero_active_when_no_fast_weights() -> None:
    controller = FastWeightController()
    tactics = ("alpha", "bravo")
    result = controller.constrain(fast_weights=None, tactic_ids=tactics)

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
