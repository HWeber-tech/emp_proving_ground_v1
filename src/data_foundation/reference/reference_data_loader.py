"""Reference data loader for instruments, sessions, and holidays."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo
import json

DAY_NAME_TO_INDEX: Mapping[str, int] = {
    "MON": 0,
    "TUE": 1,
    "WED": 2,
    "THU": 3,
    "FRI": 4,
    "SAT": 5,
    "SUN": 6,
}

INDEX_TO_DAY = {v: k.title() for k, v in DAY_NAME_TO_INDEX.items()}


@dataclass(frozen=True)
class InstrumentRecord:
    """Canonical instrument description."""

    symbol: str
    description: str
    asset_class: str
    currency: str
    tick_size: Decimal
    lot_size: Decimal
    venue: str
    session: str

    def normalised_symbol(self) -> str:
        return self.symbol.upper().strip()


@dataclass(frozen=True)
class TradingSession:
    """Trading session definition with timezone aware helpers."""

    session_id: str
    venue: str
    timezone: ZoneInfo
    open_time: time
    close_time: time
    trading_days: tuple[int, ...]
    description: str | None = None

    def is_trading_day(self, session_date: date) -> bool:
        return session_date.weekday() in self.trading_days

    def window_for_date(self, session_date: date) -> tuple[datetime, datetime]:
        start = datetime.combine(session_date, self.open_time, tzinfo=self.timezone)
        end = datetime.combine(session_date, self.close_time, tzinfo=self.timezone)
        if end <= start:
            end = end + timedelta(days=1)
        return start, end

    def contains(self, timestamp: datetime) -> bool:
        ts = timestamp.astimezone(self.timezone)
        start, end = self.window_for_date(ts.date())
        if ts < start:
            prev_start, prev_end = self.window_for_date(ts.date() - timedelta(days=1))
            return prev_start <= ts < prev_end
        if ts >= end:
            next_start, next_end = self.window_for_date(ts.date() + timedelta(days=1))
            return next_start <= ts < next_end
        return True


@dataclass(frozen=True)
class HolidayRecord:
    """Holiday record for venues/calendars."""

    calendar: str
    date: date
    name: str
    venues: tuple[str, ...]

    def applies_to(self, venue: str | None = None) -> bool:
        if venue is None or not self.venues:
            return True
        return venue.upper() in {v.upper() for v in self.venues}


class ReferenceDataLoader:
    """Load reference datasets from ``config/reference_data``."""

    def __init__(
        self,
        base_path: str | Path | None = None,
        *,
        instrument_file: str = "instruments.json",
        session_file: str = "sessions.json",
        holiday_file: str = "holidays.json",
    ) -> None:
        self._base_path = self._resolve_base_path(base_path)
        self._instrument_file = instrument_file
        self._session_file = session_file
        self._holiday_file = holiday_file
        self._instrument_cache: dict[str, InstrumentRecord] | None = None
        self._session_cache: dict[str, TradingSession] | None = None
        self._holiday_cache: tuple[HolidayRecord, ...] | None = None

    @staticmethod
    def _resolve_base_path(base_path: str | Path | None) -> Path:
        if base_path is not None:
            path = Path(base_path)
            if not path.exists():
                raise FileNotFoundError(f"Reference data directory not found: {path}")
            return path
        repo_root = Path(__file__).resolve().parents[3]
        default_path = repo_root / "config" / "reference_data"
        if default_path.exists():
            return default_path
        cwd_path = Path.cwd() / "config" / "reference_data"
        if cwd_path.exists():
            return cwd_path
        return default_path

    def instruments(self) -> tuple[InstrumentRecord, ...]:
        if self._instrument_cache is None:
            self._instrument_cache = {
                record.normalised_symbol(): record
                for record in self._load_instruments()
            }
        return tuple(self._instrument_cache.values())

    def sessions(self) -> tuple[TradingSession, ...]:
        if self._session_cache is None:
            self._session_cache = {
                session.session_id: session for session in self._load_sessions()
            }
        return tuple(self._session_cache.values())

    def holidays(self) -> tuple[HolidayRecord, ...]:
        if self._holiday_cache is None:
            self._holiday_cache = tuple(self._load_holidays())
        return self._holiday_cache

    def get_instrument(self, symbol: str) -> InstrumentRecord | None:
        if not symbol:
            return None
        key = symbol.upper().strip()
        _ = self.instruments()
        return self._instrument_cache.get(key) if self._instrument_cache else None

    def get_session(self, session_id: str) -> TradingSession | None:
        if not session_id:
            return None
        _ = self.sessions()
        return self._session_cache.get(session_id) if self._session_cache else None

    def session_for_instrument(self, symbol: str) -> TradingSession | None:
        instrument = self.get_instrument(symbol)
        if instrument is None:
            return None
        return self.get_session(instrument.session)

    def holidays_for_date(
        self,
        holiday_date: date,
        *,
        venue: str | None = None,
    ) -> tuple[HolidayRecord, ...]:
        matched = [
            holiday
            for holiday in self.holidays()
            if holiday.date == holiday_date and holiday.applies_to(venue)
        ]
        return tuple(matched)

    def is_holiday(
        self,
        holiday_date: date,
        *,
        venue: str | None = None,
    ) -> bool:
        return bool(self.holidays_for_date(holiday_date, venue=venue))

    def refresh(self) -> None:
        self._instrument_cache = None
        self._session_cache = None
        self._holiday_cache = None

    def _load_instruments(self) -> Sequence[InstrumentRecord]:
        raw_records = self._load_json_array(self._instrument_file)
        records: list[InstrumentRecord] = []
        for raw in raw_records:
            symbol = self._require_str(raw, "symbol")
            description = str(raw.get("description", "")).strip()
            asset_class = self._require_str(raw, "asset_class")
            currency = self._require_str(raw, "currency")
            tick_size = self._to_decimal(raw.get("tick_size", "1"), "tick_size")
            lot_size = self._to_decimal(raw.get("lot_size", "1"), "lot_size")
            venue = self._require_str(raw, "venue")
            session = self._require_str(raw, "session")
            records.append(
                InstrumentRecord(
                    symbol=symbol,
                    description=description,
                    asset_class=asset_class,
                    currency=currency,
                    tick_size=tick_size,
                    lot_size=lot_size,
                    venue=venue,
                    session=session,
                )
            )
        return records

    def _load_sessions(self) -> Sequence[TradingSession]:
        raw_records = self._load_json_array(self._session_file)
        sessions: list[TradingSession] = []
        for raw in raw_records:
            session_id = self._require_str(raw, "session_id")
            venue = self._require_str(raw, "venue")
            tz_name = str(raw.get("timezone", "UTC"))
            timezone = ZoneInfo(tz_name)
            open_time = self._parse_time(self._require_str(raw, "open_time"))
            close_time = self._parse_time(self._require_str(raw, "close_time"))
            raw_days = raw.get("trading_days", DAY_NAME_TO_INDEX.keys())
            trading_days = tuple(
                sorted({self._parse_day(day) for day in raw_days})
            )
            description = str(raw.get("description", "")).strip() or None
            sessions.append(
                TradingSession(
                    session_id=session_id,
                    venue=venue,
                    timezone=timezone,
                    open_time=open_time,
                    close_time=close_time,
                    trading_days=trading_days,
                    description=description,
                )
            )
        return sessions

    def _load_holidays(self) -> Sequence[HolidayRecord]:
        raw_records = self._load_json_array(self._holiday_file)
        holidays: list[HolidayRecord] = []
        for raw in raw_records:
            calendar = self._require_str(raw, "calendar")
            date_value = self._require_str(raw, "date")
            name = self._require_str(raw, "name")
            venues_raw = raw.get("venues", [])
            if isinstance(venues_raw, str):
                venues: Iterable[str] = [venues_raw]
            else:
                venues = [str(item) for item in venues_raw]
            holidays.append(
                HolidayRecord(
                    calendar=calendar,
                    date=date.fromisoformat(date_value),
                    name=name,
                    venues=tuple(venue.upper() for venue in venues),
                )
            )
        return holidays

    def _load_json_array(self, file_name: str) -> Sequence[Mapping[str, object]]:
        path = self._base_path / file_name
        if not path.exists():
            raise FileNotFoundError(f"Reference data file missing: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"Expected list in reference data file: {path}")
        typed: list[Mapping[str, object]] = []
        for item in payload:
            if not isinstance(item, Mapping):
                raise ValueError(f"Invalid record in {path}: {item!r}")
            typed.append(item)
        return typed

    @staticmethod
    def _require_str(payload: Mapping[str, object], key: str) -> str:
        value = payload.get(key)
        if value is None:
            raise ValueError(f"Missing required field '{key}' in reference data record")
        string = str(value).strip()
        if not string:
            raise ValueError(f"Field '{key}' cannot be empty in reference data record")
        return string

    @staticmethod
    def _to_decimal(value: object, field_name: str) -> Decimal:
        try:
            return Decimal(str(value))
        except (InvalidOperation, TypeError) as exc:
            raise ValueError(f"Invalid decimal for field '{field_name}': {value!r}") from exc

    @staticmethod
    def _parse_time(value: str) -> time:
        parts = value.split(":")
        if len(parts) not in {2, 3}:
            raise ValueError(f"Invalid time format: {value!r}")
        hour = int(parts[0])
        minute = int(parts[1])
        second = int(parts[2]) if len(parts) == 3 else 0
        return time(hour=hour, minute=minute, second=second)

    @staticmethod
    def _parse_day(value: object) -> int:
        if isinstance(value, int):
            if value < 0 or value > 6:
                raise ValueError(f"Trading day index must be between 0 and 6: {value}")
            return value
        if not isinstance(value, str):
            raise ValueError(f"Unsupported trading day value: {value!r}")
        key = value.strip().upper()[:3]
        if key not in DAY_NAME_TO_INDEX:
            raise ValueError(f"Unknown trading day: {value!r}")
        return DAY_NAME_TO_INDEX[key]


__all__ = [
    "InstrumentRecord",
    "TradingSession",
    "HolidayRecord",
    "ReferenceDataLoader",
]
