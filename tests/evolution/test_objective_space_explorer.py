from __future__ import annotations

import pandas as pd

from src.evolution.optimization import ObjectiveSpaceExplorer, TradeOffRecord


def _build_sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"strategy": "A", "return": 0.10, "risk": 0.05},
            {"strategy": "B", "return": 0.12, "risk": 0.07},
            {"strategy": "C", "return": 0.09, "risk": 0.04},
            {"strategy": "D", "return": 0.11, "risk": 0.06},
            {"strategy": "E", "return": 0.13, "risk": 0.05},
        ]
    )


def test_pareto_front_filters_dominated_strategies() -> None:
    frame = _build_sample_frame()
    explorer = ObjectiveSpaceExplorer(
        frame,
        ["return", "risk"],
        maximise={"return": True, "risk": False},
        metadata=["strategy"],
    )

    pareto = explorer.pareto_front()

    assert set(pareto["strategy"]) == {"C", "E"}


def test_plot_objective_space_highlights_pareto_front() -> None:
    frame = _build_sample_frame()
    explorer = ObjectiveSpaceExplorer(
        frame,
        ["return", "risk"],
        maximise={"return": True, "risk": False},
        metadata=["strategy"],
    )

    figure = explorer.plot_objective_space(title="Objectives")

    assert len(figure.data) == 2
    assert {trace.name for trace in figure.data} == {"Population", "Pareto front"}


def test_trade_off_analysis_reports_pairwise_statistics() -> None:
    frame = _build_sample_frame()
    explorer = ObjectiveSpaceExplorer(
        frame,
        ["return", "risk"],
        maximise={"return": True, "risk": False},
        metadata=["strategy"],
    )

    analysis = explorer.trade_off_analysis()

    assert len(analysis) == 1
    record = analysis[0]
    assert isinstance(record, TradeOffRecord)
    assert record.objective_a == "return"
    assert record.objective_b == "risk"
    assert record.orientation == "mixed"
    assert record.range_ratio is not None
