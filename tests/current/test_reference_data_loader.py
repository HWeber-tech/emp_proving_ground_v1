from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import pytest

from src.data_foundation.reference import ReferenceDataLoader


@pytest.fixture(scope="module")
def loader() -> ReferenceDataLoader:
    return ReferenceDataLoader()


def test_instruments_loaded(loader: ReferenceDataLoader) -> None:
    instruments = {instrument.symbol: instrument for instrument in loader.instruments()}
    assert "EURUSD" in instruments
    eurusd = instruments["EURUSD"]
    assert eurusd.tick_size == Decimal("0.00001")
    assert eurusd.lot_size == Decimal("100000")
    assert eurusd.session == "ICM_FX_DAY"


def test_session_contains_and_window(loader: ReferenceDataLoader) -> None:
    session = loader.get_session("ICM_FX_DAY")
    assert session is not None
    trading_day = date(2025, 1, 7)
    start, end = session.window_for_date(trading_day)
    assert start.tzinfo is not None
    assert start.hour == 0 and start.minute == 5
    assert end.hour == 23 and end.minute == 55

    in_window = datetime(2025, 1, 7, 14, 0, tzinfo=start.tzinfo)
    assert session.contains(in_window)

    overnight = datetime(2025, 1, 6, 23, 30, tzinfo=start.tzinfo)
    assert session.contains(overnight)

    weekend = datetime(2025, 1, 5, 12, 0, tzinfo=start.tzinfo)
    assert not session.is_trading_day(weekend.date())


def test_holiday_lookup(loader: ReferenceDataLoader) -> None:
    new_years = date(2025, 1, 1)
    assert loader.is_holiday(new_years, venue="ICM")
    assert loader.is_holiday(new_years, venue="LSE")

    random_day = date(2025, 1, 8)
    assert not loader.is_holiday(random_day, venue="ICM")

    boxing_day = date(2025, 12, 26)
    assert loader.is_holiday(boxing_day, venue="LSE")
    assert not loader.is_holiday(boxing_day, venue="ICM")

