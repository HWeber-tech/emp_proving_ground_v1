"""Helpers for constructing lightweight data slices for quick evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class DataSlice:
    symbols: List[str]
    start: datetime
    end: datetime

    def to_payload(self) -> Dict[str, object]:
        return {
            "symbols": list(self.symbols),
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "days": (self.end - self.start).days,
        }


def make_slice(
    symbols: Iterable[str],
    days: int,
    end_ts: Optional[datetime] = None,
) -> Dict[str, object]:
    """Return a simple dictionary payload describing a data slice.

    Parameters
    ----------
    symbols:
        Iterable of instrument identifiers.
    days:
        Number of calendar days to include in the slice.
    end_ts:
        Optional end timestamp. Defaults to ``datetime.utcnow()``.
    """

    cleaned_symbols = [sym for sym in (s.strip() for s in symbols) if sym]
    if not cleaned_symbols:
        raise ValueError("At least one symbol is required for a data slice")
    if days <= 0:
        raise ValueError("Slice days must be positive")

    end = end_ts or datetime.now(timezone.utc)
    start = end - timedelta(days=int(days))
    slice_obj = DataSlice(symbols=cleaned_symbols, start=start, end=end)
    return slice_obj.to_payload()


__all__ = ["make_slice", "DataSlice"]
