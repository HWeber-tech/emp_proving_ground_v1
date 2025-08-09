from datetime import datetime
from src.sensory.dimensions.what.volatility_engine import vol_signal


def test_vol_signal_basic():
    daily = [0.0005, -0.0003, 0.0002] * 200  # synthetic
    rv_win = [0.0001, -0.00005, 0.00008, -0.00002]
    vs = vol_signal("EURUSD", datetime.utcnow(), rv_win, daily)
    assert 0.0 < vs.sigma_ann < 1.0
    assert vs.regime in ("calm", "normal", "storm")

