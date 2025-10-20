import pytest

from src.core.instrument import Instrument


def test_calculate_pip_value_rejects_non_positive_price() -> None:
    instrument = Instrument.forex("EURUSD")
    with pytest.raises(ValueError):
        instrument.calculate_pip_value(0.0)
    with pytest.raises(ValueError):
        instrument.calculate_pip_value(-1.0)


def test_calculate_margin_rejects_non_positive_leverage() -> None:
    instrument = Instrument.forex("EURUSD")
    with pytest.raises(ValueError):
        instrument.calculate_margin(price=1.0, lot_size=1.0, leverage=0.0)
    with pytest.raises(ValueError):
        instrument.calculate_margin(price=1.0, lot_size=1.0, leverage=-10.0)
