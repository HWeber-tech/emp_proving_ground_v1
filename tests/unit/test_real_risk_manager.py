#!/usr/bin/env python3

from decimal import Decimal

from src.risk.real_risk_manager import RealRiskManager, RealRiskConfig
from src.core import Instrument


def test_portfolio_exposure_limits():
    cfg = RealRiskConfig(
        max_risk_per_trade_pct=Decimal('0.02'),
        max_leverage=Decimal('5.0'),
        max_total_exposure_pct=Decimal('0.1'),  # 10%
        max_drawdown_pct=Decimal('0.25'),
        min_position_size=Decimal('1000'),
        max_position_size=Decimal('1000000'),
        kelly_fraction=Decimal('0.25'),
    )
    rm = RealRiskManager(cfg)
    rm.update_account_balance(Decimal('100000'))

    # Add a position worth 5% of balance
    rm.add_position('EURUSD', Decimal('5000'), Decimal('1'))

    # Next position that would exceed 10% exposure should be invalid
    instrument = Instrument(symbol='EURUSD', pip_size=Decimal('0.0001'), lot_size=Decimal('100000'))
    is_valid = rm.validate_position(Decimal('6000'), instrument, rm.account_balance)
    assert is_valid is False


