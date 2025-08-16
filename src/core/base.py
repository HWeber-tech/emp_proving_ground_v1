"""
Compatibility shim for legacy imports used in tests.

Exposes DimensionalReading and MarketData via 'core.base', bridging to the
canonical implementations in src.sensory.organs.dimensions.base_organ.

Also provides a 'value' property on DimensionalReading as an alias to
'signal_strength' to satisfy legacy usage in tests, and a MarketData
wrapper that accepts legacy constructor fields (timestamp, bid, ask, volume, volatility)
and maps them to the canonical Pydantic model.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Type, Union, cast

# Static typing declarations for analyzers
if TYPE_CHECKING:  # pragma: no cover
    from src.sensory.organs.dimensions.base_organ import (
        DimensionalReading as DimensionalReadingDecl,
        InstrumentMeta as InstrumentMetaDecl,
        MarketData as MarketDataDecl,
        MarketRegime as MarketRegimeDecl,
    )
else:
    class MarketRegimeDecl(Enum):
        UNKNOWN = "unknown"

    class DimensionalReadingDecl:  # minimal placeholder
        ...

    class MarketDataDecl:  # minimal placeholder
        ...

    class InstrumentMetaDecl:  # minimal placeholder
        ...


# Runtime imports of canonical models
try:
    from src.sensory.organs.dimensions.base_organ import (
        DimensionalReading as _RealDimensionalReading,
        InstrumentMeta as _RealInstrumentMeta,
        MarketData as _RealMarketData,
        MarketRegime as _RealMarketRegime,
    )
except Exception:  # pragma: no cover
    # Defensive fallbacks (types already provided above)
    _RealDimensionalReading = DimensionalReadingDecl  # type: ignore[assignment]
    _RealInstrumentMeta = InstrumentMetaDecl  # type: ignore[assignment]
    _RealMarketData = MarketDataDecl  # type: ignore[assignment]
    _RealMarketRegime = MarketRegimeDecl  # type: ignore[assignment]


class DimensionalReading(_RealDimensionalReading):  # type: ignore[misc]
    @property
    def value(self) -> float:
        """
        Legacy alias expected by some tests; maps to canonical 'signal_strength'.
        """
        try:
            return float(getattr(self, "signal_strength"))
        except Exception:
            return 0.0


class MarketData(_RealMarketData):  # type: ignore[misc]
    """
    Wrapper that accepts legacy constructor arguments and maps them to the canonical model.

    Supported legacy fields:
      - timestamp: datetime
      - bid: float
      - ask: float
      - volume: float
      - volatility: float (ignored; accepted for compatibility)
      - price: float (used if bid/ask missing to infer OHLC)
      - open/high/low/close: optional; inferred when absent
      - symbol: optional; defaults to "UNKNOWN"
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Pop and normalize legacy fields first (ignore unknowns gracefully)
        symbol: str = str(kwargs.pop("symbol", "UNKNOWN"))

        # Time
        ts_raw: Any = kwargs.pop("timestamp", datetime.utcnow())
        timestamp: datetime = ts_raw if isinstance(ts_raw, datetime) else datetime.utcnow()

        # Prices
        val_price: Optional[Union[float, int, str]] = kwargs.pop("price", None)
        val_bid: Optional[Union[float, int, str]] = kwargs.pop("bid", None)
        val_ask: Optional[Union[float, int, str]] = kwargs.pop("ask", None)

        # Convert to floats with safe fallbacks
        def _to_float(x: Optional[Union[float, int, str]], default: float = 0.0) -> float:
            if x is None:
                return default
            try:
                return float(cast(Union[float, int, str], x))
            except (TypeError, ValueError):
                return default

        bid: float = _to_float(val_bid)
        ask: float = _to_float(val_ask, default=bid if bid else 0.0)

        # Close inferred from mid or provided fields
        close: float = _to_float(kwargs.pop("close", None))
        if close == 0.0:
            if bid or ask:
                close = (bid + ask) / 2.0
            else:
                close = _to_float(val_price, default=0.0)

        # OHLC fallbacks
        open_: float = _to_float(kwargs.pop("open", None), default=close)
        high: float = _to_float(kwargs.pop("high", None), default=max(open_, close))
        low: float = _to_float(kwargs.pop("low", None), default=min(open_, close))

        # Volume
        volume: float = _to_float(kwargs.pop("volume", None), default=0.0)

        # Accept and ignore legacy extras
        kwargs.pop("volatility", None)  # accepted but not used

        super().__init__(
            symbol=symbol,
            timestamp=timestamp,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            bid=bid,
            ask=ask,
            **kwargs,
        )


# Re-export canonical enums and types with precise static types
MarketRegime: Type[MarketRegimeDecl] = _RealMarketRegime  # type: ignore[assignment]
InstrumentMeta: Type[InstrumentMetaDecl] = _RealInstrumentMeta  # type: ignore[assignment]

__all__ = ["DimensionalReading", "MarketData", "MarketRegime", "InstrumentMeta"]