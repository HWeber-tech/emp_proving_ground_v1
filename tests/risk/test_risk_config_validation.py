from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.config.risk.risk_config import RiskConfig


def test_min_position_size_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        RiskConfig(min_position_size=0)


def test_min_position_size_cannot_exceed_maximum() -> None:
    with pytest.raises(ValidationError):
        RiskConfig(min_position_size=2_000, max_position_size=1_000)


def test_max_exposure_must_cover_risk_per_trade() -> None:
    with pytest.raises(ValidationError):
        RiskConfig(
            max_risk_per_trade_pct=Decimal("0.20"),
            max_total_exposure_pct=Decimal("0.10"),
        )


def test_stop_loss_disabled_outside_research_mode_emits_warning() -> None:
    message = "mandatory_stop_loss=False outside research mode"
    with pytest.warns(UserWarning, match=message):
        RiskConfig(mandatory_stop_loss=False, research_mode=False)


def test_stop_loss_can_be_disabled_in_research_mode() -> None:
    cfg = RiskConfig(mandatory_stop_loss=False, research_mode=True)
    assert cfg.research_mode is True
    assert cfg.mandatory_stop_loss is False
