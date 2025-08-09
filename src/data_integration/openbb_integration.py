from __future__ import annotations

"""
OpenBB integration for macro/economic data (WHY) and yields.
Falls back gracefully if OpenBB is not installed.
"""

from typing import List, Optional
from datetime import datetime

try:
    from openbb import obb  # type: ignore
except Exception:  # pragma: no cover
    obb = None

from src.data_foundation.schemas import MacroEvent, YieldEvent


def fetch_calendar(start: str, end: str, importance: Optional[str] = None) -> List[MacroEvent]:
    """Fetch macro calendar events via OpenBB, return canonical MacroEvent list."""
    if obb is None:
        return []
    events: List[MacroEvent] = []
    try:
        df = obb.macro.calendar(start_date=start, end_date=end)  # type: ignore[attr-defined]
        # Optionally filter by importance if column exists
        for _, row in df.iterrows():
            try:
                imp = str(row.get("importance") or "")
                if importance and imp.lower() != importance.lower():
                    continue
                events.append(MacroEvent(
                    timestamp=datetime.fromisoformat(str(row.get("date"))),
                    calendar="openbb",
                    event=str(row.get("event")),
                    currency=str(row.get("currency") or ""),
                    actual=_to_float(row.get("actual")),
                    forecast=_to_float(row.get("forecast")),
                    previous=_to_float(row.get("previous")),
                    importance=imp,
                    source="openbb",
                ))
            except Exception:
                continue
    except Exception:
        return []
    return events


def _to_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def fetch_yields(curve: str = "UST") -> list[YieldEvent]:
    """Fetch yield curve points via OpenBB; returns canonical YieldEvent list."""
    if obb is None:
        return []
    events: list[YieldEvent] = []
    try:
        df = obb.fixedincome.yield_curve(curve=curve)  # type: ignore[attr-defined]
        ts = datetime.utcnow()
        for col in df.columns:
            try:
                val = _to_float(df[col].iloc[-1])
                if val is None:
                    continue
                events.append(YieldEvent(timestamp=ts, curve=curve, tenor=str(col), value=val, source="openbb"))
            except Exception:
                continue
    except Exception:
        return []
    return events


