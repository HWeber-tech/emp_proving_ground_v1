from decimal import Decimal

import pytest

from src.config.risk.risk_config import RiskConfig
from src.runtime.predator_app import _resolve_risk_config


def test_resolve_risk_config_respects_extras_overrides() -> None:
    extras = {
        "RISK_MAX_TOTAL_EXPOSURE_PCT": "0.25",
        "RISK_MAX_LEVERAGE": "5.5",
        "RISK_MIN_POSITION_SIZE": "5000",
        "RISK_MAX_POSITION_SIZE": "750000",
        "RISK_MANDATORY_STOP_LOSS": "false",
        "RISK_RESEARCH_MODE": "true",
    }

    config = _resolve_risk_config(extras, risk_per_trade=0.05, max_drawdown=0.12)

    assert isinstance(config, RiskConfig)
    assert config.max_risk_per_trade_pct == Decimal("0.05")
    assert config.max_total_exposure_pct == Decimal("0.25")
    assert config.max_leverage == Decimal("5.5")
    assert config.max_drawdown_pct == Decimal("0.12")
    assert config.min_position_size == 5000
    assert config.max_position_size == 750000
    assert config.mandatory_stop_loss is False
    assert config.research_mode is True


@pytest.mark.parametrize(
    "extras",
    [
        {"RISK_MIN_POSITION_SIZE": "abc", "RISK_MAX_POSITION_SIZE": ""},
        {"RISK_MANDATORY_STOP_LOSS": "maybe", "RISK_RESEARCH_MODE": "-"},
        {"RISK_MAX_LEVERAGE": "invalid", "RISK_MAX_TOTAL_EXPOSURE_PCT": "not-a-number"},
    ],
)
def test_resolve_risk_config_falls_back_on_invalid_extras(extras: dict[str, str]) -> None:
    config = _resolve_risk_config(extras, risk_per_trade=0.02, max_drawdown=0.1)

    assert config.min_position_size == 1
    assert config.max_position_size == 1_000_000
    assert config.mandatory_stop_loss is True
    assert config.research_mode is False
    assert config.max_leverage == Decimal("10.0")
    assert config.max_total_exposure_pct == Decimal("0.5")
