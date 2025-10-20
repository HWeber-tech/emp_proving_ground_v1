from __future__ import annotations

import pytest

from src.trading.risk import estimate_l1_queue_fill_probability


def test_queue_fill_probability_scales_with_share_and_flow() -> None:
    probability = estimate_l1_queue_fill_probability(
        our_size=5,
        queue_size=20,
        trade_flow_factor=0.6,
        queue_smoothing=5,
    )

    assert probability == pytest.approx(0.12)


def test_queue_fill_probability_clamps_to_unit_interval() -> None:
    probability = estimate_l1_queue_fill_probability(
        our_size=100,
        queue_size=1,
        trade_flow_factor=2.0,
    )

    assert probability == pytest.approx(1.0)


def test_queue_fill_probability_handles_missing_inputs() -> None:
    assert estimate_l1_queue_fill_probability(None, None, None) == 0.0
    assert estimate_l1_queue_fill_probability(10, 0, 0.5, queue_smoothing=0) == 0.0
