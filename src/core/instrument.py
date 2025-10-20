"""
Instrument Module
=================

Defines the Instrument class for financial instruments.
"""

from __future__ import annotations

from dataclasses import dataclass
try:  # Python 3.10 compatibility for typing.Unpack
    from typing import TypedDict, Unpack, cast
except ImportError:  # pragma: no cover - fallback for older runtimes
    from typing import TypedDict, cast

    try:
        from typing_extensions import Unpack  # type: ignore
    except ImportError as exc:  # pragma: no cover - make failure explicit during import
        raise ImportError("typing.Unpack requires Python 3.11 or typing_extensions") from exc

from src.core.coercion import coerce_float
from src.core.types import JSONObject


class InstrumentPayload(TypedDict, total=False):
    symbol: str
    name: str
    instrument_type: str
    base_currency: str
    quote_currency: str
    pip_value: float
    contract_size: float
    min_lot_size: float
    max_lot_size: float
    tick_size: float


class _InstrumentCtorKwargs(TypedDict, total=False):
    base_currency: str
    quote_currency: str
    pip_value: float
    contract_size: float
    min_lot_size: float
    max_lot_size: float
    tick_size: float


class _CryptoCtorKwargs(TypedDict, total=False):
    base_currency: str
    quote_currency: str
    contract_size: float
    min_lot_size: float
    max_lot_size: float
    tick_size: float


__all__ = [
    "Instrument",
    "InstrumentPayload",
    "get_instrument",
    "get_all_instruments",
    "FOREX_INSTRUMENTS",
]


def _coerce_to_float(value: object, default: float) -> float:
    """Best-effort coercion of value to float; falls back to default on failure."""
    coerced = coerce_float(value, default=default)
    return default if coerced is None else coerced


@dataclass
class Instrument:
    """Represents a financial instrument."""

    symbol: str
    name: str
    instrument_type: str = "forex"
    base_currency: str = ""
    quote_currency: str = ""
    pip_value: float = 0.0001
    contract_size: float = 100000.0
    min_lot_size: float = 0.01
    max_lot_size: float = 100.0
    tick_size: float = 0.00001

    def __post_init__(self) -> None:
        """Initialize derived attributes."""
        if not self.base_currency and self.symbol:
            # Extract base currency from symbol (e.g., EUR from EURUSD)
            self.base_currency = self.symbol[:3]
            self.quote_currency = self.symbol[3:]

    def calculate_pip_value(self, price: float, lot_size: float = 1.0) -> float:
        """Calculate pip value for given price and lot size."""
        if price <= 0:
            raise ValueError("price must be positive")
        return (self.pip_value * self.contract_size * lot_size) / price

    def calculate_margin(self, price: float, lot_size: float, leverage: float = 100.0) -> float:
        """Calculate required margin for position."""
        if price <= 0:
            raise ValueError("price must be positive")
        if leverage <= 0:
            raise ValueError("leverage must be positive")
        position_value = abs(lot_size) * self.contract_size * price
        return position_value / leverage

    def to_dict(self) -> InstrumentPayload:
        """Convert instrument to dictionary."""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "instrument_type": self.instrument_type,
            "base_currency": self.base_currency,
            "quote_currency": self.quote_currency,
            "pip_value": self.pip_value,
            "contract_size": self.contract_size,
            "min_lot_size": self.min_lot_size,
            "max_lot_size": self.max_lot_size,
            "tick_size": self.tick_size,
        }

    @classmethod
    def from_dict(cls, data: InstrumentPayload | JSONObject) -> Instrument:
        """Create instrument from dictionary.

        Performs safe coercion of numeric fields that may be provided as str|int|float.
        Leaves unknown keys untouched to preserve current error semantics.
        """
        # Make a shallow copy so we can adjust values without mutating the caller's dict
        payload: dict[str, object] = dict(data)

        # Coerce known numeric fields if present
        if "pip_value" in payload:
            payload["pip_value"] = _coerce_to_float(payload.get("pip_value"), cls.pip_value)
        if "contract_size" in payload:
            payload["contract_size"] = _coerce_to_float(
                payload.get("contract_size"), cls.contract_size
            )
        if "min_lot_size" in payload:
            payload["min_lot_size"] = _coerce_to_float(
                payload.get("min_lot_size"), cls.min_lot_size
            )
        if "max_lot_size" in payload:
            payload["max_lot_size"] = _coerce_to_float(
                payload.get("max_lot_size"), cls.max_lot_size
            )
        if "tick_size" in payload:
            payload["tick_size"] = _coerce_to_float(payload.get("tick_size"), cls.tick_size)

        return cls(**cast(InstrumentPayload, payload))

    @classmethod
    def forex(cls, symbol: str, **kwargs: Unpack[_InstrumentCtorKwargs]) -> Instrument:
        """Create forex instrument."""
        return cls(symbol=symbol, name=f"{symbol} Forex", instrument_type="forex", **kwargs)

    @classmethod
    def crypto(cls, symbol: str, **kwargs: Unpack[_CryptoCtorKwargs]) -> Instrument:
        """Create crypto instrument."""
        return cls(
            symbol=symbol,
            name=f"{symbol} Crypto",
            instrument_type="crypto",
            pip_value=0.01,
            **kwargs,
        )


# Common forex instruments
FOREX_INSTRUMENTS: dict[str, Instrument] = {
    "EURUSD": Instrument.forex("EURUSD"),
    "GBPUSD": Instrument.forex("GBPUSD"),
    "USDJPY": Instrument.forex("USDJPY"),
    "USDCHF": Instrument.forex("USDCHF"),
    "AUDUSD": Instrument.forex("AUDUSD"),
    "USDCAD": Instrument.forex("USDCAD"),
    "NZDUSD": Instrument.forex("NZDUSD"),
    "EURGBP": Instrument.forex("EURGBP"),
    "EURJPY": Instrument.forex("EURJPY"),
    "GBPJPY": Instrument.forex("GBPJPY"),
}


def get_instrument(symbol: str | None) -> Instrument | None:
    """Get instrument by symbol."""
    if symbol is None:
        return None
    normalised = "".join(ch for ch in symbol.strip().upper() if ch.isalnum())
    if not normalised:
        return None
    return FOREX_INSTRUMENTS.get(normalised)


def get_all_instruments() -> dict[str, Instrument]:
    """Get all available instruments."""
    return FOREX_INSTRUMENTS.copy()


if __name__ == "__main__":
    # Test instrument creation
    eurusd = get_instrument("EURUSD")
    if eurusd:
        print(f"EURUSD: {eurusd}")
        print(f"Pip value at 1.10: {eurusd.calculate_pip_value(1.10)}")
        print(f"Margin for 1 lot at 1.10: {eurusd.calculate_margin(1.10, 1.0)}")
