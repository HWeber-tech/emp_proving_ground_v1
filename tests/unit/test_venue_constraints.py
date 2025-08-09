from src.operational.venue_constraints import align_quantity, align_price, normalize_tif


def test_align_quantity_enforces_min_and_step():
    assert align_quantity('EURUSD', 100) == 1000
    assert align_quantity('EURUSD', 1500) == 1000
    assert align_quantity('EURUSD', 2500) == 2000


def test_align_price_ticks():
    assert align_price('EURUSD', 1.1234567) == 1.12346
    assert align_price('USDJPY', 157.1239) == 157.124


def test_normalize_tif_defaults_to_day():
    assert normalize_tif('0') == '0'
    assert normalize_tif('999') == '0'

