"""Load canonical reference data aligned with the high-impact roadmap."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

_ASSET_CLASS_VALUES: frozenset[str] = frozenset({"equity", "fx_fut", "fx_spot"})
_VENUE_VALUES: frozenset[str] = frozenset({"nasdaq", "globex", "spot_agg"})


__all__ = [
    "InstrumentDefinition",
    "TradingSession",
    "Holiday",
    "ReferenceDataSet",
    "ReferenceDataLoader",
]


# ---------------------------------------------------------------------------
# Data models


@dataclass(frozen=True, slots=True)
class InstrumentDefinition:
    """Describes a tradeable instrument and core risk parameters."""

    symbol: str
    contract_size: Decimal
    pip_decimal_places: int
    margin_currency: str
    asset_class: str | None = None
    venue: str | None = None
    long_swap_rate: Decimal | None = None
    short_swap_rate: Decimal | None = None
    swap_time: time | None = None


@dataclass(frozen=True, slots=True)
class TradingSession:
    """Represents a trading session window for a venue or geography."""

    name: str
    timezone: str
    open_time: time
    close_time: time
    days: tuple[str, ...]
    venue: str | None = None
    description: str | None = None


@dataclass(frozen=True, slots=True)
class Holiday:
    """Holiday entry indicating when venues are closed or observing half-days."""

    date: date
    name: str
    venues: tuple[str, ...]
    description: str | None = None


@dataclass(frozen=True, slots=True)
class ReferenceDataSet:
    """Bundle of reference artefacts consumed by downstream components."""

    instruments: Mapping[str, InstrumentDefinition]
    sessions: Mapping[str, TradingSession]
    holidays: tuple[Holiday, ...]


# ---------------------------------------------------------------------------
# Loader implementation


class ReferenceDataLoader:
    """Load reference data (instruments, sessions, holidays) from disk."""

    _SUPPORTED_SUFFIXES = (".json", ".yaml", ".yml")

    def __init__(
        self,
        *,
        config_root: Path | str | None = None,
        instrument_path: Path | str | None = None,
        session_path: Path | str | None = None,
        holiday_path: Path | str | None = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[3]
        default_root = project_root / "config"
        self._config_root = Path(config_root) if config_root is not None else default_root

        self._instrument_path = self._resolve_path(
            instrument_path,
            [self._config_root / "system" / "instruments.json"],
        )
        self._session_path = self._resolve_path(
            session_path,
            [self._config_root / "reference" / "sessions"],
        )
        self._holiday_path = self._resolve_path(
            holiday_path,
            [self._config_root / "reference" / "holidays"],
        )

        self._instrument_cache: Mapping[str, InstrumentDefinition] | None = None
        self._session_cache: Mapping[str, TradingSession] | None = None
        self._holiday_cache: tuple[Holiday, ...] | None = None

    # ------------------------------------------------------------------
    def load_instruments(self, *, refresh: bool = False) -> Mapping[str, InstrumentDefinition]:
        """Return instrument definitions keyed by symbol."""

        if not refresh and self._instrument_cache is not None:
            return self._instrument_cache

        payload = self._read_structured(self._instrument_path)
        if not isinstance(payload, Mapping):
            raise TypeError("Instrument config must contain a mapping of symbol -> payload")

        instruments: Dict[str, InstrumentDefinition] = {}
        for symbol, raw in payload.items():
            if not isinstance(raw, Mapping):
                continue
            try:
                instruments[str(symbol).upper()] = self._parse_instrument(symbol, raw)
            except Exception as exc:  # pragma: no cover - defensive guard
                raise ValueError(f"Invalid instrument payload for {symbol!r}: {exc}") from exc

        self._instrument_cache = instruments
        return instruments

    def load_sessions(self, *, refresh: bool = False) -> Mapping[str, TradingSession]:
        """Return trading sessions keyed by identifier."""

        if not refresh and self._session_cache is not None:
            return self._session_cache

        payload = self._read_structured(self._session_path)
        if not isinstance(payload, Mapping):
            raise TypeError("Session config must contain a mapping of id -> payload")

        sessions: Dict[str, TradingSession] = {}
        for key, raw in payload.items():
            if not isinstance(raw, Mapping):
                continue
            try:
                sessions[str(key)] = self._parse_session(raw)
            except Exception as exc:  # pragma: no cover - defensive guard
                raise ValueError(f"Invalid session payload for {key!r}: {exc}") from exc

        self._session_cache = sessions
        return sessions

    def load_holidays(self, *, refresh: bool = False) -> tuple[Holiday, ...]:
        """Return holiday entries sorted chronologically."""

        if not refresh and self._holiday_cache is not None:
            return self._holiday_cache

        payload = self._read_structured(self._holiday_path)
        if isinstance(payload, Mapping):
            records: Iterable[Mapping[str, object]] = payload.values()  # pragma: no cover - alt schema
        elif isinstance(payload, Sequence):
            records = [item for item in payload if isinstance(item, Mapping)]
        else:
            raise TypeError("Holiday config must be a list or mapping of entries")

        holidays: list[Holiday] = []
        for entry in records:
            try:
                holidays.append(self._parse_holiday(entry))
            except Exception as exc:  # pragma: no cover - defensive guard
                raise ValueError(f"Invalid holiday payload {entry}: {exc}") from exc

        holidays.sort(key=lambda item: item.date)
        self._holiday_cache = tuple(holidays)
        return self._holiday_cache

    def load_all(self, *, refresh: bool = False) -> ReferenceDataSet:
        """Return a combined reference dataset."""

        instruments = self.load_instruments(refresh=refresh)
        sessions = self.load_sessions(refresh=refresh)
        holidays = self.load_holidays(refresh=refresh)
        return ReferenceDataSet(
            instruments=instruments,
            sessions=sessions,
            holidays=holidays,
        )

    # ------------------------------------------------------------------
    def _resolve_path(
        self,
        explicit: Path | str | None,
        candidates: Sequence[Path],
    ) -> Path:
        if explicit is not None:
            explicit_path = Path(explicit)
            if not explicit_path.exists():
                raise FileNotFoundError(f"Reference data file not found: {explicit_path}")
            return explicit_path

        for candidate in candidates:
            if candidate.exists():
                return candidate
            for suffix in self._SUPPORTED_SUFFIXES:
                alt = candidate.with_suffix(suffix)
                if alt.exists():
                    return alt
        raise FileNotFoundError(
            f"Reference data file not found. Tried: {', '.join(str(path) for path in candidates)}"
        )

    @staticmethod
    def _read_structured(path: Path) -> object:
        data = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:  # pragma: no cover - optional dependency
                import yaml  # type: ignore

                return yaml.safe_load(data)
            except Exception as exc:  # pragma: no cover - degrade gracefully
                raise RuntimeError(f"Unable to parse YAML reference data: {path}") from exc
        return json.loads(data)

    @staticmethod
    def _parse_decimal(value: object | None) -> Decimal | None:
        if value is None:
            return None
        return Decimal(str(value))

    @staticmethod
    def _parse_time(value: object | None) -> time | None:
        if value in (None, "", False):
            return None
        text = str(value)
        for fmt in ("%H:%M", "%H:%M:%S"):
            try:
                return datetime.strptime(text, fmt).time()
            except ValueError:
                continue
        raise ValueError(f"Invalid time format: {text}")

    @staticmethod
    def _normalise_enum(
        value: object | None,
        allowed: frozenset[str],
        *,
        field: str,
        symbol: str,
    ) -> str | None:
        if value is None:
            return None
        candidate = str(value).strip().lower()
        if not candidate:
            return None
        if candidate not in allowed:
            options = ", ".join(sorted(allowed))
            raise ValueError(f"Invalid {field} {value!r} for {symbol}; expected one of {options}")
        return candidate

    def _parse_instrument(self, symbol: str, payload: Mapping[str, object]) -> InstrumentDefinition:
        contract_size = self._parse_decimal(payload.get("contract_size")) or Decimal("0")
        pip_places = int(payload.get("pip_decimal_places", 0))
        margin_currency = str(payload.get("margin_currency", "")).upper()
        long_swap = self._parse_decimal(payload.get("long_swap_rate"))
        short_swap = self._parse_decimal(payload.get("short_swap_rate"))
        swap_time = self._parse_time(payload.get("swap_time"))
        asset_class = self._normalise_enum(
            payload.get("asset_class"),
            _ASSET_CLASS_VALUES,
            field="asset_class",
            symbol=str(symbol),
        )
        venue = self._normalise_enum(
            payload.get("venue"),
            _VENUE_VALUES,
            field="venue",
            symbol=str(symbol),
        )

        return InstrumentDefinition(
            symbol=str(symbol).upper(),
            contract_size=contract_size,
            pip_decimal_places=pip_places,
            margin_currency=margin_currency,
            asset_class=asset_class,
            venue=venue,
            long_swap_rate=long_swap,
            short_swap_rate=short_swap,
            swap_time=swap_time,
        )

    def _parse_session(self, payload: Mapping[str, object]) -> TradingSession:
        name = str(payload.get("name", "")).strip()
        timezone_name = str(payload.get("timezone", "UTC"))
        open_time = self._parse_time(payload.get("open_time"))
        close_time = self._parse_time(payload.get("close_time"))
        if open_time is None or close_time is None:
            raise ValueError("Session requires open_time and close_time")

        raw_days = payload.get("days") or ("Mon", "Tue", "Wed", "Thu", "Fri")
        if isinstance(raw_days, Sequence) and not isinstance(raw_days, (str, bytes)):
            days: tuple[str, ...] = tuple(str(day) for day in raw_days)
        else:
            days = (str(raw_days),)

        venue = payload.get("venue")
        description = payload.get("description")

        return TradingSession(
            name=name or "Session",
            timezone=timezone_name,
            open_time=open_time,
            close_time=close_time,
            days=days,
            venue=str(venue) if venue is not None else None,
            description=str(description) if description is not None else None,
        )

    def _parse_holiday(self, payload: Mapping[str, object]) -> Holiday:
        raw_date = payload.get("date")
        if isinstance(raw_date, date):
            holiday_date = raw_date
        elif isinstance(raw_date, datetime):
            holiday_date = raw_date.date()
        else:
            holiday_date = datetime.fromisoformat(str(raw_date)).date()

        venues_field = payload.get("venues") or payload.get("markets")
        venues: tuple[str, ...]
        if isinstance(venues_field, Sequence) and not isinstance(venues_field, (str, bytes)):
            venues = tuple(str(venue).upper() for venue in venues_field)
        elif venues_field:
            venues = (str(venues_field).upper(),)
        else:
            venues = tuple()

        description = payload.get("description")

        return Holiday(
            date=holiday_date,
            name=str(payload.get("name", "")),
            venues=venues,
            description=str(description) if description is not None else None,
        )
