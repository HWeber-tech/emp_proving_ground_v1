#!/usr/bin/env python3

from decimal import Decimal

from src.risk.real_risk_manager import RealRiskManager, RealRiskConfig


class DummyBus:
    def __init__(self):
        self.events = []

    def publish_sync(self, event_type, data, source=None):
        self.events.append((event_type, data, source))


def test_risk_manager_emits_event_on_breach():
    cfg = RealRiskConfig(max_var_pct=Decimal('0.01'), max_es_pct=Decimal('0.02'))
    rm = RealRiskManager(cfg)
    bus = DummyBus()
    rm.event_bus = bus

    rm.positions = {
        'A': {'size': Decimal('10000'), 'value': 10000.0, 'return': -0.05},
        'B': {'size': Decimal('10000'), 'value': 10000.0, 'return': -0.04},
    }

    allowed = rm.check_risk_thresholds()
    assert allowed is False
    assert any(e[0] == 'risk_threshold_exceeded' for e in bus.events)


