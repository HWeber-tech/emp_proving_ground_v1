#!/usr/bin/env python3

from decimal import Decimal

from src.risk.real_risk_manager import RealRiskManager, RealRiskConfig


def test_var_es_thresholds_enforced():
    cfg = RealRiskConfig(
        max_risk_per_trade_pct=Decimal('0.02'),
        max_leverage=Decimal('5.0'),
        max_total_exposure_pct=Decimal('0.5'),
        max_drawdown_pct=Decimal('0.25'),
        min_position_size=Decimal('1000'),
        max_position_size=Decimal('1000000'),
        kelly_fraction=Decimal('0.25'),
        max_var_pct=Decimal('0.01'),  # 1%
        max_es_pct=Decimal('0.02'),   # 2%
    )
    rm = RealRiskManager(cfg)
    rm.update_account_balance(Decimal('100000'))

    # Create positions with negative returns to breach ES/VAR
    rm.positions = {
        'A': {'size': Decimal('10000'), 'value': 10000.0, 'return': -0.03},
        'B': {'size': Decimal('10000'), 'value': 10000.0, 'return': -0.025},
        'C': {'size': Decimal('10000'), 'value': 10000.0, 'return': -0.02},
        'D': {'size': Decimal('10000'), 'value': 10000.0, 'return': -0.015},
        'E': {'size': Decimal('10000'), 'value': 10000.0, 'return': -0.01},
    }

    within_limits = rm.check_risk_thresholds()
    assert within_limits is False


