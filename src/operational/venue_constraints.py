"""
Venue constraints and normalization utilities for IC Markets cTrader FIX.

- Min volume: 1000 units, step 1000 (micro lot)
- Tick size: 0.00001 for most FX; 0.001 for JPY-quoted pairs
- Allowed TIF: Day(0), IOC(3), FOK(4), GTD(6)
"""

from __future__ import annotations

from typing import Tuple


def get_min_volume(symbol: str) -> int:
    return 1000


def get_volume_step(symbol: str) -> int:
    return 1000


def align_quantity(symbol: str, quantity: float) -> int:
    step = get_volume_step(symbol)
    q = int(quantity // step * step)
    if q < get_min_volume(symbol):
        q = get_min_volume(symbol)
    return q


def is_jpy_quoted(symbol: str) -> bool:
    s = (symbol or "").upper()
    return s.endswith("JPY")


def get_tick_size(symbol: str) -> float:
    return 0.001 if is_jpy_quoted(symbol) else 0.00001


def align_price(symbol: str, price: float) -> float:
    tick = get_tick_size(symbol)
    # Avoid float issues: scale to int, round, then scale back
    scaled = round(price / tick)
    return scaled * tick


def allowed_tif() -> Tuple[str, ...]:
    return ("0", "3", "4", "6")


def normalize_tif(value: str) -> str:
    v = str(value)
    if v not in allowed_tif():
        return "0"
    return v


