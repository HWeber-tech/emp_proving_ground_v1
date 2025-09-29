"""Order book imbalance analytics for the HOW sensory dimension."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from src.trading.order_management.order_book.snapshot import (
    OrderBookLevel,
    OrderBookSnapshot,
)

__all__ = ["OrderBookImbalanceMetrics", "compute_order_book_imbalance"]


@dataclass(frozen=True)
class OrderBookImbalanceMetrics:
    """Summary statistics describing order book imbalance."""

    buy_volume: float
    sell_volume: float
    total_volume: float
    imbalance: float
    levels_evaluated: int

    @property
    def has_volume(self) -> bool:
        return self.total_volume > 0.0


def compute_order_book_imbalance(
    order_book: OrderBookSnapshot | Mapping[str, object] | None,
    *,
    depth: int | None = None,
) -> OrderBookImbalanceMetrics:
    """Calculate an order flow imbalance score for a snapshot.

    The function accepts either the canonical :class:`OrderBookSnapshot` model
    or any mapping containing ``"bids"``/``"asks"`` sequences.  Each level can be
    represented as an :class:`OrderBookLevel`, mapping (``{"price": ..., "volume": ...}``)
    or a tuple ``(price, volume)``.  Levels are assumed to be ordered from best
    to worst, matching FIX book updates.

    Args:
        order_book: Snapshot or mapping describing the order book levels.
        depth: Optional number of levels per side to include in the calculation.

    Returns:
        ``OrderBookImbalanceMetrics`` describing aggregated buy/sell volume and
        a normalised imbalance in ``[-1, 1]`` where positive numbers indicate bid
        pressure.
    """

    bids, asks = _extract_levels(order_book)

    if depth is not None and depth > 0:
        bids = bids[:depth]
        asks = asks[:depth]

    buy_volume = sum(volume for _, volume in bids)
    sell_volume = sum(volume for _, volume in asks)
    total_volume = buy_volume + sell_volume

    if total_volume <= 0.0:
        return OrderBookImbalanceMetrics(
            buy_volume=0.0,
            sell_volume=0.0,
            total_volume=0.0,
            imbalance=0.0,
            levels_evaluated=len(bids) + len(asks),
        )

    imbalance = (buy_volume - sell_volume) / total_volume
    imbalance = max(min(imbalance, 1.0), -1.0)

    return OrderBookImbalanceMetrics(
        buy_volume=float(buy_volume),
        sell_volume=float(sell_volume),
        total_volume=float(total_volume),
        imbalance=float(imbalance),
        levels_evaluated=len(bids) + len(asks),
    )


def _extract_levels(
    order_book: OrderBookSnapshot | Mapping[str, object] | None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    if isinstance(order_book, OrderBookSnapshot):
        return (
            _normalise_levels(order_book.bids),
            _normalise_levels(order_book.asks),
        )

    if isinstance(order_book, Mapping):
        bids = order_book.get("bids")
        asks = order_book.get("asks")
        return (
            _normalise_levels(bids if isinstance(bids, Iterable) else ()),
            _normalise_levels(asks if isinstance(asks, Iterable) else ()),
        )

    return ([], [])


def _normalise_levels(levels: Iterable[object]) -> list[tuple[float, float]]:
    normalised: list[tuple[float, float]] = []
    for level in levels:
        if isinstance(level, OrderBookLevel):
            price = float(level.price)
            volume = float(level.volume)
        elif isinstance(level, Mapping):
            price = float(level.get("price", 0.0) or 0.0)
            volume = float(level.get("volume", 0.0) or 0.0)
        elif isinstance(level, Sequence) and len(level) >= 2:
            price = float(level[0])
            volume = float(level[1])
        else:
            continue

        if volume <= 0.0:
            continue

        normalised.append((price, volume))

    return normalised

