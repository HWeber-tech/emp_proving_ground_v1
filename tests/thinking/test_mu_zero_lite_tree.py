from __future__ import annotations

import pytest

from src.thinking.evaluation.muzero_lite_tree import (
    MuZeroLitePath,
    MuZeroLiteTreeResult,
    simulate_short_horizon_futures,
)


def test_simulate_short_horizon_futures_evaluates_paths() -> None:
    policy = {
        "root": {"buy": 0.6, "sell": 0.4},
        "s1": {"hold": 1.0},
        "s2": {"hold": 1.0},
        "terminal": {},
    }
    transitions = {
        "root": {
            "buy": {"state": "s1", "edge": 4.0, "causal": {"liquidity": 0.5}},
            "sell": {"state": "s2", "edge": -1.0, "causal": ["macro"]},
        },
        "s1": {"hold": ("terminal", 2.0)},
        "s2": {"hold": ("terminal", 1.0)},
    }
    values = {"terminal": 0.5}

    result = simulate_short_horizon_futures(
        "root",
        policy=policy,
        transition_model=transitions,
        value_model=values,
        horizon=2,
        discount=0.9,
        causal_edge_adjustments={"liquidity": 1.0, "macro": -0.5},
    )

    assert isinstance(result, MuZeroLiteTreeResult)
    assert result.best_path is not None
    assert result.best_path.actions == ("buy", "hold")
    assert result.best_path.total_return == pytest.approx(6.705, rel=1e-9)

    paths = {path.actions: path for path in result.paths}
    assert set(paths.keys()) == {("buy", "hold"), ("sell", "hold")}
    assert paths[("sell", "hold")].total_return == pytest.approx(-0.195, rel=1e-9)

    assert result.expected_return == pytest.approx(3.945, rel=1e-9)

    payload = result.as_dict()
    assert payload["best_path"]["actions"] == ["buy", "hold"]
    assert len(payload["paths"]) == 2


def test_simulate_short_horizon_requires_positive_horizon() -> None:
    policy = {"root": {"buy": 1.0}}
    transitions = {"root": {"buy": ("terminal", 1.0)}}

    with pytest.raises(ValueError):
        simulate_short_horizon_futures(
            "root",
            policy=policy,
            transition_model=transitions,
            value_model={"terminal": 0.0},
            horizon=0,
        )


def test_simulate_short_horizon_handles_absorbing_state() -> None:
    policy = {"root": {}}

    result = simulate_short_horizon_futures(
        "root",
        policy=policy,
        transition_model={},
        value_model={"root": 1.2},
        horizon=1,
    )

    assert len(result.paths) == 1
    path = result.paths[0]
    assert isinstance(path, MuZeroLitePath)
    assert path.steps == tuple()
    assert path.total_return == pytest.approx(1.2, rel=1e-9)


def test_simulate_short_horizon_respects_max_branches() -> None:
    policy = {
        "root": {"a": 0.5, "b": 0.3, "c": 0.2},
        "terminal": {},
    }
    transitions = {
        "root": {
            "a": ("terminal", 1.0),
            "b": ("terminal", 2.0),
            "c": ("terminal", 3.0),
        }
    }

    result = simulate_short_horizon_futures(
        "root",
        policy=policy,
        transition_model=transitions,
        value_model={"terminal": 0.0},
        horizon=1,
        max_branches=2,
    )

    assert len(result.paths) == 2
    actions = {path.actions[0] for path in result.paths if path.actions}
    assert actions == {"a", "b"}
