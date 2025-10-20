from __future__ import annotations

from datetime import date, time
from pathlib import Path

import json
import pytest

from src.data_foundation.reference import ReferenceDataLoader


def test_default_reference_loader_reads_repo_configs() -> None:
    loader = ReferenceDataLoader()

    instruments = loader.load_instruments()
    assert "EURUSD" in instruments
    eurusd = instruments["EURUSD"]
    assert float(eurusd.contract_size) == pytest.approx(100000)
    assert eurusd.pip_decimal_places == 4
    assert eurusd.swap_time == time(hour=22)
    assert eurusd.asset_class == "fx_spot"
    assert eurusd.venue == "spot_agg"

    assert {"6E_DEC24", "AAPL"}.issubset(instruments.keys())
    futures = instruments["6E_DEC24"]
    assert futures.asset_class == "fx_fut"
    assert futures.venue == "globex"
    equity = instruments["AAPL"]
    assert equity.asset_class == "equity"
    assert equity.venue == "nasdaq"

    asset_classes = {inst.asset_class for inst in instruments.values() if inst.asset_class}
    assert asset_classes == {"equity", "fx_fut", "fx_spot"}
    venues = {inst.venue for inst in instruments.values() if inst.venue}
    assert venues == {"nasdaq", "globex", "spot_agg"}

    sessions = loader.load_sessions()
    assert set(sessions.keys()) >= {"london_fx", "tokyo_fx", "new_york_fx"}
    london = sessions["london_fx"]
    assert london.open_time == time(hour=7)
    assert london.close_time == time(hour=16, minute=30)
    assert london.days == ("Mon", "Tue", "Wed", "Thu", "Fri")

    holidays = loader.load_holidays()
    assert holidays[0].date == date(2025, 1, 1)
    assert {"LONDON", "NEW_YORK", "TOKYO"}.issubset(set(holidays[0].venues))


def test_loader_refresh_reads_modified_files(tmp_path: Path) -> None:
    instrument_path = tmp_path / "instruments.json"
    sessions_path = tmp_path / "sessions.json"
    holidays_path = tmp_path / "holidays.json"

    instrument_path.write_text(
        json.dumps(
            {
                "TESTFX": {
                    "contract_size": "1000",
                    "pip_decimal_places": 3,
                    "margin_currency": "USD",
                    "swap_time": "21:00",
                }
            }
        ),
        encoding="utf-8",
    )
    sessions_path.write_text(
        json.dumps(
            {
                "test": {
                    "name": "Test Session",
                    "timezone": "UTC",
                    "open_time": "01:00",
                    "close_time": "02:00",
                    "days": ["Mon"],
                }
            }
        ),
        encoding="utf-8",
    )
    holidays_path.write_text(
        json.dumps(
            [
                {
                    "date": "2025-05-01",
                    "name": "Test Holiday",
                    "venues": ["TEST"],
                }
            ]
        ),
        encoding="utf-8",
    )

    loader = ReferenceDataLoader(
        instrument_path=instrument_path,
        session_path=sessions_path,
        holiday_path=holidays_path,
    )

    instruments = loader.load_instruments()
    assert set(instruments.keys()) == {"TESTFX"}

    # Modify the instrument file and ensure cached call still returns original until refresh
    instrument_path.write_text(
        json.dumps(
            {
                "TESTFX": {
                    "contract_size": "2000",
                    "pip_decimal_places": 3,
                    "margin_currency": "USD",
                    "swap_time": "22:00",
                }
            }
        ),
        encoding="utf-8",
    )

    cached = loader.load_instruments()
    assert float(cached["TESTFX"].contract_size) == pytest.approx(1000)

    refreshed = loader.load_instruments(refresh=True)
    assert float(refreshed["TESTFX"].contract_size) == pytest.approx(2000)

    holidays = loader.load_holidays(refresh=True)
    assert holidays == loader.load_holidays()


def test_loader_rejects_missing_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        ReferenceDataLoader(instrument_path=missing)
