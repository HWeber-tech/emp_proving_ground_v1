"""Order book analytics feeding the HOW sensory dimension.

The high-impact roadmap calls for richer execution-aligned telemetry such as
order book imbalance scores and volume profile snapshots.  The helpers in this
module normalise heterogeneous order book inputs (live FIX snapshots, mock
feeds, DataFrame columns) into a consistent analytic payload that can be
consumed by sensors and risk tooling without introducing heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import tanh
from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "OrderBookMetrics",
    "OrderBookSideProfile",
    "compute_order_book_metrics",
]


@dataclass(slots=True)
class OrderBookSideProfile:
    """Summary of a single order book side for telemetry purposes."""

    side: str
    price: float
    volume: float
    share_of_side: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "side": self.side,
            "price": self.price,
            "volume": self.volume,
            "share_of_side": self.share_of_side,
        }


@dataclass(slots=True)
class OrderBookMetrics:
    """Computed analytics for a given order book snapshot."""

    total_bid_volume: float
    total_ask_volume: float
    imbalance: float
    spread: float
    mid_price: float
    top_of_book_liquidity: float
    depth_liquidity: float
    volume_profile: list[OrderBookSideProfile]

    def as_payload(self) -> dict[str, float | list[dict[str, float | str]]]:
        """Return a mapping suitable for sensor payload enrichment."""

        return {
            "order_imbalance": self.imbalance,
            "book_spread": self.spread,
            "book_mid_price": self.mid_price,
            "book_depth": self.depth_liquidity,
            "top_of_book_liquidity": self.top_of_book_liquidity,
            "volume_profile": [profile.as_dict() for profile in self.volume_profile],
        }


def _coerce_levels(
    levels: Sequence[object] | Iterable[object],
    *,
    descending: bool,
) -> tuple[np.ndarray, np.ndarray]:
    prices: list[float] = []
    volumes: list[float] = []

    for level in levels:
        price: float | None = None
        volume: float | None = None

        if hasattr(level, "price") and hasattr(level, "volume"):
            price = float(getattr(level, "price"))
            volume = float(getattr(level, "volume"))
        elif isinstance(level, Mapping):
            if "price" in level and "volume" in level:
                price = float(level["price"])
                volume = float(level["volume"])
            elif "price" in level and "size" in level:
                price = float(level["price"])
                volume = float(level["size"])
            elif "price" in level and "qty" in level:
                price = float(level["price"])
                volume = float(level["qty"])
        else:
            try:
                price = float(level[0])  # type: ignore[index]
                volume = float(level[1])  # type: ignore[index]
            except (TypeError, IndexError):  # pragma: no cover - defensive
                price = None
                volume = None

        if price is None or volume is None:
            continue

        if volume <= 0:
            continue

        prices.append(price)
        volumes.append(volume)

    if not prices:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    order = np.argsort(prices)
    if descending:
        order = order[::-1]
    prices_array = np.asarray(prices, dtype=float)[order]
    volumes_array = np.asarray(volumes, dtype=float)[order]
    return prices_array, volumes_array


def _extract_levels(book: object) -> tuple[Sequence[object], Sequence[object]]:
    if book is None:
        return (), ()

    if hasattr(book, "bids") and hasattr(book, "asks"):
        bids = getattr(book, "bids")
        asks = getattr(book, "asks")
        return bids or (), asks or ()

    if isinstance(book, Mapping):
        bids = book.get("bids") or book.get("bid_levels") or ()
        asks = book.get("asks") or book.get("ask_levels") or ()
        return bids, asks

    if isinstance(book, Sequence) and len(book) == 2:
        return book[0], book[1]

    return (), ()


def compute_order_book_metrics(
    book: object,
    *,
    depth_normaliser: float = 50_000.0,
    top_levels: int = 3,
) -> OrderBookMetrics:
    """Compute imbalance and liquidity metrics for an order book snapshot."""

    bids_raw, asks_raw = _extract_levels(book)
    bid_prices, bid_volumes = _coerce_levels(bids_raw, descending=True)
    ask_prices, ask_volumes = _coerce_levels(asks_raw, descending=False)

    total_bid_volume = float(bid_volumes.sum()) if bid_volumes.size else 0.0
    total_ask_volume = float(ask_volumes.sum()) if ask_volumes.size else 0.0
    depth_liquidity = total_bid_volume + total_ask_volume

    if bid_prices.size and ask_prices.size:
        spread = float(max(ask_prices[0] - bid_prices[0], 0.0))
        mid_price = float((ask_prices[0] + bid_prices[0]) / 2.0)
        top_of_book_liquidity = float(bid_volumes[0] + ask_volumes[0])
    else:
        spread = 0.0
        mid_price = 0.0
        top_of_book_liquidity = float(bid_volumes[0] if bid_volumes.size else 0.0)
        top_of_book_liquidity += float(ask_volumes[0] if ask_volumes.size else 0.0)

    imbalance_denominator = total_bid_volume + total_ask_volume
    if imbalance_denominator <= 0:
        imbalance = 0.0
    else:
        imbalance = float(
            (total_bid_volume - total_ask_volume)
            / max(imbalance_denominator, 1e-9)
        )
        imbalance = float(max(-1.0, min(1.0, imbalance)))

    # Liquidity score is a soft clamp of overall depth to [0, 1]
    depth_scale = max(depth_normaliser, 1.0)
    depth_score = float(tanh(depth_liquidity / depth_scale))

    volume_profile: list[OrderBookSideProfile] = []
    if bid_volumes.size:
        total = total_bid_volume or 1.0
        for price, volume in zip(bid_prices[:top_levels], bid_volumes[:top_levels]):
            volume_profile.append(
                OrderBookSideProfile(
                    side="bid",
                    price=float(price),
                    volume=float(volume),
                    share_of_side=float(volume / total),
                )
            )
    if ask_volumes.size:
        total = total_ask_volume or 1.0
        for price, volume in zip(ask_prices[:top_levels], ask_volumes[:top_levels]):
            volume_profile.append(
                OrderBookSideProfile(
                    side="ask",
                    price=float(price),
                    volume=float(volume),
                    share_of_side=float(volume / total),
                )
            )

    return OrderBookMetrics(
        total_bid_volume=total_bid_volume,
        total_ask_volume=total_ask_volume,
        imbalance=imbalance,
        spread=spread,
        mid_price=mid_price,
        top_of_book_liquidity=top_of_book_liquidity,
        depth_liquidity=depth_score,
        volume_profile=volume_profile,
    )
