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


def test_instrument_from_dict_coerces_grouped_numeric_strings() -> None:
    instrument = Instrument.from_dict(
        {
            "symbol": "EURUSD",
            "name": "Euro / US Dollar",
            "pip_value": "0.000 5",
            "contract_size": "1 000 000",
            "min_lot_size": "0.5",
            "max_lot_size": "2 500",
            "tick_size": "0.000 05",
        }
    )

    assert instrument.contract_size == 1_000_000.0
    assert instrument.max_lot_size == 2500.0
    assert instrument.pip_value == pytest.approx(0.0005)
    assert instrument.tick_size == pytest.approx(0.00005)
