from __future__ import annotations

from decimal import Decimal

import pytest

import src.risk as risk


def test_position_sizing_helpers_compute_expected_values() -> None:
    fraction = risk.kelly_fraction(win_rate=0.55, avg_win=0.02, avg_loss=0.01)
    size = risk.position_size(
        balance=Decimal("100000"),
        risk_per_trade=Decimal("0.02"),
        stop_loss_pct=Decimal("0.01"),
    )

    assert 0.0 < fraction < 1.0
    assert size == Decimal("200000.0")


def test_risk_module_dynamic_exports_are_cached() -> None:
    cls = risk.RiskManager
    again = risk.RiskManager

    assert cls is again
    assert cls.__name__ == "RiskManager"

    with pytest.raises(AttributeError):
        risk.__getattr__("does_not_exist")


def test_quantile_helpers_extract_and_scale() -> None:
    triplet = risk.normalise_quantile_triplet({"q25": -0.01, "q50": 0.02, "q75": 0.05})
    assert triplet == pytest.approx((-0.01, 0.02, 0.05))

    ratio = risk.quantile_edge_ratio({"0.25": -0.01, "0.50": 0.02, "0.75": 0.05})

    assert ratio == pytest.approx(0.02 / 0.06)
