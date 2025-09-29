"""Order book analytics utilities for the HOW sensory dimension.

The high-impact roadmap calls out richer microstructure analytics so the
trading stack can reason about venue liquidity and imbalance in real time.
This module provides a lightweight transformer that summarises level-two
snapshots into structured metrics which can be consumed by sensors,
strategies, and risk components without requiring a heavy data science
dependency chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

__all__ = [
    "OrderBookAnalyticsConfig",
    "OrderBookSnapshot",
    "OrderBookAnalytics",
]


_BID_COLUMN_ALIASES = {"bid_price", "bid", "bid_px"}
_ASK_COLUMN_ALIASES = {"ask_price", "ask", "ask_px"}
_BID_SIZE_ALIASES = {"bid_size", "bid_qty", "bid_quantity"}
_ASK_SIZE_ALIASES = {"ask_size", "ask_qty", "ask_quantity"}


@dataclass(slots=True)
class OrderBookAnalyticsConfig:
    """Configuration for order book analytics."""

    depth_levels: int = 5
    """Maximum number of depth levels to include in calculations."""

    value_area_fraction: float = 0.7
    """Fraction of volume that defines the *value area* for the profile."""

    imbalance_smoothing: float = 1e-6
    """Epsilon to avoid division by zero when computing ratios."""


@dataclass(slots=True)
class OrderBookSnapshot:
    """Container for structured order book analytics."""

    imbalance: float
    spread: float
    mid_price: float
    value_area_low: float
    value_area_high: float
    total_bid_volume: float
    total_ask_volume: float
    participation_ratio: float

    def as_dict(self) -> dict[str, float]:
        return {
            "imbalance": float(self.imbalance),
            "spread": float(self.spread),
            "mid_price": float(self.mid_price),
            "value_area_low": float(self.value_area_low),
            "value_area_high": float(self.value_area_high),
            "total_bid_volume": float(self.total_bid_volume),
            "total_ask_volume": float(self.total_ask_volume),
            "participation_ratio": float(self.participation_ratio),
        }


class OrderBookAnalytics:
    """Summarise level-two snapshots into microstructure metrics."""

    def __init__(self, config: OrderBookAnalyticsConfig | None = None) -> None:
        self._config = config or OrderBookAnalyticsConfig()

    def describe(self, order_book: pd.DataFrame | None) -> OrderBookSnapshot | None:
        """Produce an :class:`OrderBookSnapshot` from raw depth data.

        The analytics gracefully handle partially populated snapshots and
        return ``None`` when insufficient information is available.
        """

        if order_book is None or order_book.empty:
            return None

        frame = order_book.copy()

        price_columns = self._select_first_existing(frame.columns, _BID_COLUMN_ALIASES)
        ask_price_columns = self._select_first_existing(
            frame.columns, _ASK_COLUMN_ALIASES
        )
        bid_size_columns = self._select_first_existing(frame.columns, _BID_SIZE_ALIASES)
        ask_size_columns = self._select_first_existing(frame.columns, _ASK_SIZE_ALIASES)

        if not price_columns or not ask_price_columns or not bid_size_columns or not ask_size_columns:
            return None

        depth = min(len(frame), max(1, self._config.depth_levels))
        frame = frame.iloc[:depth]

        bid_prices = frame[price_columns[0]].astype(float)
        ask_prices = frame[ask_price_columns[0]].astype(float)
        bid_sizes = frame[bid_size_columns[0]].astype(float).clip(lower=0.0)
        ask_sizes = frame[ask_size_columns[0]].astype(float).clip(lower=0.0)

        total_bid = float(bid_sizes.sum())
        total_ask = float(ask_sizes.sum())
        denom = total_bid + total_ask + self._config.imbalance_smoothing
        imbalance = (total_bid - total_ask) / denom

        best_bid = float(bid_prices.iloc[0])
        best_ask = float(ask_prices.iloc[0])
        spread = max(0.0, best_ask - best_bid)
        mid = best_bid + spread / 2 if spread >= 0 else (best_bid + best_ask) / 2

        profile_prices = np.concatenate([bid_prices.to_numpy(), ask_prices.to_numpy()])
        profile_volumes = np.concatenate([bid_sizes.to_numpy(), ask_sizes.to_numpy()])
        value_area_low, value_area_high = self._value_area(profile_prices, profile_volumes)

        participation_ratio = float(total_bid / denom)

        return OrderBookSnapshot(
            imbalance=float(imbalance),
            spread=float(spread),
            mid_price=float(mid),
            value_area_low=float(value_area_low),
            value_area_high=float(value_area_high),
            total_bid_volume=total_bid,
            total_ask_volume=total_ask,
            participation_ratio=participation_ratio,
        )

    def _value_area(
        self, prices: np.ndarray, volumes: np.ndarray
    ) -> tuple[float, float]:
        if prices.size == 0 or volumes.size == 0:
            return (0.0, 0.0)

        positive_volumes = np.clip(volumes, a_min=0.0, a_max=None)
        total_volume = positive_volumes.sum()
        if total_volume <= 0:
            price = float(np.mean(prices)) if prices.size else 0.0
            return (price, price)

        weights = positive_volumes / total_volume
        sorted_idx = np.argsort(prices)
        sorted_prices = prices[sorted_idx]
        sorted_weights = weights[sorted_idx]

        cumulative = np.cumsum(sorted_weights)
        value_fraction = float(min(max(self._config.value_area_fraction, 0.0), 1.0))
        lower_fraction = (1.0 - value_fraction) / 2
        upper_fraction = 1.0 - lower_fraction

        low_index = int(
            min(len(sorted_prices) - 1, np.searchsorted(cumulative, lower_fraction, side="left"))
        )
        high_index = int(
            min(len(sorted_prices) - 1, np.searchsorted(cumulative, upper_fraction, side="left"))
        )
        value_area_low = float(sorted_prices[low_index])
        value_area_high = float(sorted_prices[high_index])

        return (value_area_low, value_area_high)

    @staticmethod
    def _select_first_existing(columns: Iterable[str], candidates: set[str]) -> list[str]:
        available = [col for col in columns if col in candidates]
        return available

