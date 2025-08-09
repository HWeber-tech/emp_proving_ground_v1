from src.trading.risk.pretrade_risk import PreTradeRiskGate, RiskLimits


def test_pretrade_risk_caps():
    gate = PreTradeRiskGate(RiskLimits(max_notional=2000.0, max_volume=5000, max_exposure_per_symbol=3000))
    # Allow small buy
    assert gate.check_order("EURUSD", "1", 1000, 1.10) is True
    # Exposure grows; but still under cap
    assert gate.check_order("EURUSD", "1", 1000, 1.11) is True
    # Deny exposure breach
    assert gate.check_order("EURUSD", "1", 5000, 1.12) is False
    # Deny notional
    assert gate.check_order("EURUSD", "1", 5000, 1.0) is False
    # Volume cap
    assert gate.check_order("EURUSD", "1", 100000, 1.0) is False

