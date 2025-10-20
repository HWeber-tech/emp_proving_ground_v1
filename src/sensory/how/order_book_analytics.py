"""Order book analytics utilities for the HOW sensory dimension.

The high-impact roadmap calls out richer microstructure analytics so the
trading stack can reason about venue liquidity and imbalance in real time.
This module provides a lightweight transformer that summarises level-two
snapshots into structured metrics which can be consumed by sensors,
strategies, and risk components without requiring a heavy data science
dependency chain.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime
from math import isfinite
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "OrderBookAnalyticsConfig",
    "OrderBookSnapshot",
    "OrderBookAnalytics",
    "TickSpaceDepthEncoder",
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

    depth_embedding_dim: int = 12
    """Dimensionality of the tick-space depth embedding (8–16 inclusive)."""


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
    microprice: float
    microprice_offset: float
    spread_ticks: float
    spread_wide: float
    queue_imbalance: float
    ofi_norm: float
    refresh_hz: float
    stale_ms: float
    slope_bid: float
    slope_ask: float
    curve_bid: float
    curve_ask: float
    depth_embedding: tuple[float, ...]
    liquidity_bucket_low: float = 0.0
    liquidity_bucket_mid: float = 1.0
    liquidity_bucket_high: float = 0.0

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
            "microprice": float(self.microprice),
            "microprice_offset": float(self.microprice_offset),
            "spread_ticks": float(self.spread_ticks),
            "spread_wide": float(self.spread_wide),
            "queue_imbalance": float(self.queue_imbalance),
            "ofi_norm": float(self.ofi_norm),
            "refresh_hz": float(self.refresh_hz),
            "stale_ms": float(self.stale_ms),
            "slope_bid": float(self.slope_bid),
            "slope_ask": float(self.slope_ask),
            "curve_bid": float(self.curve_bid),
            "curve_ask": float(self.curve_ask),
            "depth_embedding": [float(value) for value in self.depth_embedding],
            "liquidity_bucket_low": float(self.liquidity_bucket_low),
            "liquidity_bucket_mid": float(self.liquidity_bucket_mid),
            "liquidity_bucket_high": float(self.liquidity_bucket_high),
        }

    @classmethod
    def scalar_fields(cls) -> tuple[str, ...]:
        return (
            "imbalance",
            "spread",
            "mid_price",
            "value_area_low",
            "value_area_high",
            "total_bid_volume",
            "total_ask_volume",
            "participation_ratio",
            "microprice",
            "microprice_offset",
            "spread_ticks",
            "spread_wide",
            "queue_imbalance",
            "ofi_norm",
            "refresh_hz",
            "stale_ms",
            "slope_bid",
            "slope_ask",
            "curve_bid",
            "curve_ask",
            "liquidity_bucket_low",
            "liquidity_bucket_mid",
            "liquidity_bucket_high",
        )

    def flatten(self, prefix: str = "order_book_") -> dict[str, float]:
        metrics: dict[str, float] = {}
        for name in self.scalar_fields():
            metrics[f"{prefix}{name}"] = float(getattr(self, name))
        for idx, value in enumerate(self.depth_embedding):
            metrics[f"{prefix}depth_embedding_{idx}"] = float(value)
        return metrics


class TickSpaceDepthEncoder:
    """Encode order book depth into a compact tick-space representation."""

    _MIN_EMBEDDING = 8
    _MAX_EMBEDDING = 16
    _DEFAULT_KERNEL_SIZE = 4

    def __init__(
        self,
        *,
        max_depth: int,
        embedding_dim: int,
        kernel_size: int | None = None,
        epsilon: float = 1e-9,
    ) -> None:
        if embedding_dim < self._MIN_EMBEDDING or embedding_dim > self._MAX_EMBEDDING:
            raise ValueError(
                f"depth embedding dimension must be between {self._MIN_EMBEDDING} and {self._MAX_EMBEDDING}"
            )
        self.max_depth = max(1, int(max_depth))
        self.embedding_dim = int(embedding_dim)
        self.kernel_size = max(1, int(kernel_size or self._DEFAULT_KERNEL_SIZE))
        self._epsilon = float(max(epsilon, 1e-12))

        # Shared convolution kernels for bids and asks (weights are deterministic).
        self._kernel_linear = np.array(
            [
                [0.58, 0.31, 0.11],
                [0.27, 0.44, 0.19],
                [0.12, 0.18, 0.29],
                [0.05, 0.08, 0.24],
            ],
            dtype=np.float64,
        )
        self._kernel_gate = np.array(
            [
                [0.21, -0.28, 0.16],
                [0.14, 0.17, -0.23],
                [0.08, 0.09, 0.19],
                [0.03, 0.05, 0.11],
            ],
            dtype=np.float64,
        )
        self._bias_linear = 0.0
        self._bias_gate = -0.1

    def encode(
        self,
        bid_prices: Sequence[float],
        bid_sizes: Sequence[float],
        ask_prices: Sequence[float],
        ask_sizes: Sequence[float],
    ) -> np.ndarray:
        """Return a tick-space embedding of the current depth snapshot."""

        level_limit = min(
            len(bid_prices), len(bid_sizes), len(ask_prices), len(ask_sizes), self.max_depth
        )
        if level_limit <= 0:
            return np.zeros(self.embedding_dim, dtype=np.float64)

        bid_prices_arr = np.asarray(bid_prices[:level_limit], dtype=np.float64)
        bid_sizes_arr = np.asarray(bid_sizes[:level_limit], dtype=np.float64)
        ask_prices_arr = np.asarray(ask_prices[:level_limit], dtype=np.float64)
        ask_sizes_arr = np.asarray(ask_sizes[:level_limit], dtype=np.float64)

        if bid_prices_arr.size == 0 or ask_prices_arr.size == 0:
            return np.zeros(self.embedding_dim, dtype=np.float64)

        tick_size = self._estimate_tick_size(bid_prices_arr, ask_prices_arr)
        if not isfinite(tick_size) or tick_size <= self._epsilon:
            reference = max(abs(float(bid_prices_arr[0])) * 1e-6, 1e-6)
            tick_size = reference

        ask_prices_arr, ask_sizes_arr = self._ensure_ask_axis_orientation(
            ask_prices_arr, ask_sizes_arr
        )

        bid_features = self._construct_features(bid_prices_arr, bid_sizes_arr, tick_size)
        ask_features = self._construct_features(ask_prices_arr, ask_sizes_arr, tick_size)

        bid_conv = self._apply_convolution(bid_features)
        ask_conv = self._apply_convolution(ask_features)

        half_dim = self.embedding_dim // 2
        bid_dim = half_dim + (self.embedding_dim % 2)
        ask_dim = half_dim

        bid_embedding = self._adaptive_pool(bid_conv, bid_dim)
        ask_embedding = self._adaptive_pool(ask_conv, ask_dim)

        concatenated = np.concatenate([bid_embedding, ask_embedding])
        return np.tanh(concatenated)

    def _estimate_tick_size(self, bids: np.ndarray, asks: np.ndarray) -> float:
        all_prices = np.concatenate([bids, asks])
        unique = np.unique(np.sort(all_prices))
        if unique.size <= 1:
            return 0.0
        diffs = np.diff(unique)
        positive_diffs = diffs[diffs > 0]
        if positive_diffs.size == 0:
            return 0.0
        return float(np.min(positive_diffs))

    def _ensure_ask_axis_orientation(
        self, prices: np.ndarray, sizes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if prices.size == 0:
            return prices, sizes
        best_idx = int(np.argmin(prices))
        if best_idx != 0:
            prices = np.roll(prices, -best_idx)
            sizes = np.roll(sizes, -best_idx)
        if prices[0] > prices[-1]:
            prices = prices[::-1]
            sizes = sizes[::-1]
        return prices, sizes

    def _construct_features(
        self, prices: np.ndarray, sizes: np.ndarray, tick_size: float
    ) -> np.ndarray:
        clipped_sizes = np.clip(sizes, a_min=0.0, a_max=None)
        level_count = prices.size
        if level_count == 0:
            return np.zeros((0, 3), dtype=np.float64)

        tick_offsets = np.abs(prices - prices[0]) / max(tick_size, self._epsilon)
        tick_scale = max(tick_offsets.max(), 1.0)
        tick_feature = tick_offsets / tick_scale

        volume_sum = clipped_sizes.sum() + self._epsilon
        volume_ratio = clipped_sizes / volume_sum
        log_volume = np.log1p(clipped_sizes)
        cumulative = np.cumsum(volume_ratio)
        density = volume_ratio / (1.0 + tick_offsets)

        feature_matrix = np.stack([tick_feature, log_volume, cumulative + density], axis=1)
        return feature_matrix

    def _apply_convolution(self, features: np.ndarray) -> np.ndarray:
        length = features.shape[0]
        if length == 0:
            return np.zeros(1, dtype=np.float64)

        kernel_size = min(self.kernel_size, length)
        kernel_lin = self._kernel_linear[:kernel_size]
        kernel_gate = self._kernel_gate[:kernel_size]

        window_count = length - kernel_size + 1
        windows = np.stack([features[i : i + kernel_size] for i in range(window_count)], axis=0)

        conv_linear = np.tensordot(windows, kernel_lin, axes=([1, 2], [0, 1])) + self._bias_linear
        conv_gate = np.tensordot(windows, kernel_gate, axes=([1, 2], [0, 1])) + self._bias_gate

        return conv_linear * self._sigmoid(conv_gate)

    def _adaptive_pool(self, values: np.ndarray, target_dim: int) -> np.ndarray:
        if target_dim <= 0:
            return np.zeros(0, dtype=np.float64)
        if values.size == 0:
            return np.zeros(target_dim, dtype=np.float64)
        if values.size == 1:
            return np.full(target_dim, float(values[0]), dtype=np.float64)

        indices = np.linspace(0, values.size - 1, target_dim)
        base = np.arange(values.size)
        pooled = np.interp(indices, base, values)
        return pooled.astype(np.float64)

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(slots=True)
class _LiquidityQuantileTracker:
    """Track daily median L1 size/spread ratios to derive liquidity buckets."""

    max_history: int = 30
    _history: deque[float] = field(init=False, repr=False)
    _current_day: date | None = field(init=False, default=None, repr=False)
    _current_values: list[float] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self) -> None:
        self._history = deque(maxlen=max(1, int(self.max_history)))

    def update(self, timestamp: datetime | None, ratio: float) -> str:
        if timestamp is None:
            return "mid"

        day = timestamp.date()
        if self._current_day is None or day != self._current_day:
            self._finalise_current_day()
            self._current_day = day
            self._current_values = []

        value = float(max(ratio, 0.0))
        if not isfinite(value):
            value = 0.0
        self._current_values.append(value)

        current_median = float(np.median(self._current_values)) if self._current_values else 0.0
        lower, upper = self._historical_quantiles()
        if lower is None or upper is None:
            return "mid"
        if current_median <= lower:
            return "low"
        if current_median >= upper:
            return "high"
        return "mid"

    def _finalise_current_day(self) -> None:
        if not self._current_values:
            return
        median_value = float(np.median(self._current_values))
        if isfinite(median_value):
            self._history.append(median_value)

    def _historical_quantiles(self) -> tuple[float | None, float | None]:
        if not self._history:
            return (None, None)
        data = np.array(self._history, dtype=float)
        if data.size == 0:
            return (None, None)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return (None, None)
        lower = float(np.quantile(finite, 1.0 / 3.0))
        upper = float(np.quantile(finite, 2.0 / 3.0))
        return (lower, upper)


@dataclass(slots=True)
class _OrderBookState:
    """Minimal information cached between observations."""

    best_bid_price: float
    best_bid_size: float
    best_ask_price: float
    best_ask_size: float
    timestamp: datetime | None


class OrderBookAnalytics:
    """Summarise level-two snapshots into microstructure metrics."""

    def __init__(
        self,
        config: OrderBookAnalyticsConfig | None = None,
        *,
        depth_encoder: TickSpaceDepthEncoder | None = None,
    ) -> None:
        self._config = config or OrderBookAnalyticsConfig()
        self._depth_encoder = depth_encoder or TickSpaceDepthEncoder(
            max_depth=self._config.depth_levels,
            embedding_dim=self._config.depth_embedding_dim,
        )
        self._metric_names: tuple[str, ...] | None = None
        self._history: dict[str, _OrderBookState] = {}
        self._liquidity_trackers: dict[str, _LiquidityQuantileTracker] = {}

    def describe(
        self,
        order_book: pd.DataFrame | None,
        *,
        symbol: str | None = None,
        timestamp: datetime | None = None,
        trade_sign: float | None = None,
    ) -> OrderBookSnapshot | None:
        """Produce an :class:`OrderBookSnapshot` from raw depth data.

        The analytics gracefully handle partially populated snapshots and
        return ``None`` when insufficient information is available.
        """

        if order_book is None or order_book.empty:
            return None

        frame = order_book.copy()
        state_key = symbol or "__default__"

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

        bid_array = bid_prices.to_numpy()
        ask_array = ask_prices.to_numpy()
        profile_prices = np.concatenate([bid_array, ask_array])
        profile_volumes = np.concatenate([bid_sizes.to_numpy(), ask_sizes.to_numpy()])
        value_area_low, value_area_high = self._value_area(profile_prices, profile_volumes)

        participation_ratio = float(total_bid / denom)

        tick_size = self._estimate_tick_size(bid_array, ask_array)
        best_bid_size = float(bid_sizes.iloc[0]) if not bid_sizes.empty else 0.0
        best_ask_size = float(ask_sizes.iloc[0]) if not ask_sizes.empty else 0.0
        queue_imbalance = self._queue_imbalance(best_bid_size, best_ask_size)
        microprice = self._microprice(best_bid, best_ask, best_bid_size, best_ask_size, mid)
        microprice_offset = microprice - mid
        spread_ticks = self._spread_in_ticks(spread, tick_size)
        spread_wide = 1.0 if spread_ticks > 1.0 + 1e-9 else 0.0

        tracker = self._liquidity_trackers.setdefault(state_key, _LiquidityQuantileTracker())
        bucket_label = tracker.update(
            timestamp,
            self._liquidity_ratio(best_bid_size, best_ask_size, spread, tick_size),
        )
        if bucket_label == "low":
            liquidity_bucket_low, liquidity_bucket_mid, liquidity_bucket_high = (1.0, 0.0, 0.0)
        elif bucket_label == "high":
            liquidity_bucket_low, liquidity_bucket_mid, liquidity_bucket_high = (0.0, 0.0, 1.0)
        else:
            liquidity_bucket_low, liquidity_bucket_mid, liquidity_bucket_high = (0.0, 1.0, 0.0)

        previous = self._history.get(state_key)
        ofi_norm = self._ofi_norm(
            previous,
            best_bid,
            best_bid_size,
            best_ask,
            best_ask_size,
            trade_sign,
        )
        refresh_hz, stale_ms = self._refresh_metrics(previous, timestamp)

        slope_bid, curve_bid = self._depth_shape_features(
            bid_array,
            bid_sizes.to_numpy(),
            tick_size,
            side="bid",
        )
        slope_ask, curve_ask = self._depth_shape_features(
            ask_array,
            ask_sizes.to_numpy(),
            tick_size,
            side="ask",
        )

        depth_embedding = tuple(
            float(value)
            for value in self._depth_encoder.encode(
                bid_array,
                bid_sizes.to_numpy(),
                ask_array,
                ask_sizes.to_numpy(),
            )
        )

        self._history[state_key] = _OrderBookState(
            best_bid_price=best_bid,
            best_bid_size=best_bid_size,
            best_ask_price=best_ask,
            best_ask_size=best_ask_size,
            timestamp=timestamp,
        )

        return OrderBookSnapshot(
            imbalance=float(imbalance),
            spread=float(spread),
            mid_price=float(mid),
            value_area_low=float(value_area_low),
            value_area_high=float(value_area_high),
            total_bid_volume=total_bid,
            total_ask_volume=total_ask,
            participation_ratio=participation_ratio,
            microprice=float(microprice),
            microprice_offset=float(microprice_offset),
            spread_ticks=float(spread_ticks),
            spread_wide=float(spread_wide),
            queue_imbalance=float(queue_imbalance),
            ofi_norm=float(ofi_norm),
            refresh_hz=float(refresh_hz),
            stale_ms=float(stale_ms),
            slope_bid=float(slope_bid),
            slope_ask=float(slope_ask),
            curve_bid=float(curve_bid),
            curve_ask=float(curve_ask),
            depth_embedding=depth_embedding,
            liquidity_bucket_low=float(liquidity_bucket_low),
            liquidity_bucket_mid=float(liquidity_bucket_mid),
            liquidity_bucket_high=float(liquidity_bucket_high),
        )

    def metric_names(self, prefix: str = "order_book_") -> tuple[str, ...]:
        if self._metric_names is None:
            base = list(OrderBookSnapshot.scalar_fields())
            base.extend(
                f"depth_embedding_{idx}" for idx in range(self._depth_encoder.embedding_dim)
            )
            self._metric_names = tuple(base)
        if not prefix:
            return self._metric_names
        return tuple(f"{prefix}{name}" for name in self._metric_names)

    def _estimate_tick_size(self, bids: np.ndarray, asks: np.ndarray) -> float:
        tick = float(self._depth_encoder._estimate_tick_size(bids, asks))
        if not isfinite(tick) or tick <= 0.0:
            reference_prices = np.concatenate([bids[:1], asks[:1]])
            reference = float(np.mean(reference_prices)) if reference_prices.size else 1.0
            tick = max(abs(reference) * 1e-6, 1e-9)
        return tick

    @staticmethod
    def _queue_imbalance(best_bid_size: float, best_ask_size: float) -> float:
        denom = best_bid_size + best_ask_size
        if denom <= 0.0:
            return 0.0
        return (best_bid_size - best_ask_size) / denom

    @staticmethod
    def _microprice(
        best_bid: float,
        best_ask: float,
        bid_size: float,
        ask_size: float,
        mid: float,
    ) -> float:
        depth = bid_size + ask_size
        if depth <= 0.0:
            return mid
        return (best_ask * bid_size + best_bid * ask_size) / depth

    @staticmethod
    def _spread_in_ticks(spread: float, tick_size: float) -> float:
        if tick_size <= 0.0:
            return 0.0
        return spread / tick_size

    def _ofi_norm(
        self,
        previous: _OrderBookState | None,
        best_bid: float,
        best_bid_size: float,
        best_ask: float,
        best_ask_size: float,
        trade_sign: float | None,
    ) -> float:
        if previous is None:
            return 0.0

        prev_bid_px = previous.best_bid_price
        prev_bid_sz = previous.best_bid_size
        prev_ask_px = previous.best_ask_price
        prev_ask_sz = previous.best_ask_size

        bid_component = 0.0
        if best_bid > prev_bid_px:
            bid_component = best_bid_size
        elif best_bid < prev_bid_px:
            bid_component = -prev_bid_sz
        else:
            bid_component = best_bid_size - prev_bid_sz

        ask_component = 0.0
        if best_ask < prev_ask_px:
            ask_component = -best_ask_size
        elif best_ask > prev_ask_px:
            ask_component = prev_ask_sz
        else:
            ask_component = prev_ask_sz - best_ask_size

        ofi = bid_component + ask_component
        sign = self._normalise_trade_sign(trade_sign)
        ofi *= sign

        scale = prev_bid_sz + prev_ask_sz
        if scale <= 0.0:
            scale = max(best_bid_size + best_ask_size, 1e-9)

        return ofi / max(scale, 1e-9)

    @staticmethod
    def _normalise_trade_sign(value: float | None) -> float:
        if value is None:
            return 1.0
        try:
            sign = float(value)
        except (TypeError, ValueError):
            return 1.0
        if sign > 0:
            return 1.0
        if sign < 0:
            return -1.0
        return 1.0

    @staticmethod
    def _refresh_metrics(
        previous: _OrderBookState | None,
        timestamp: datetime | None,
    ) -> tuple[float, float]:
        if previous is None or timestamp is None or previous.timestamp is None:
            return (0.0, 0.0)
        delta = (timestamp - previous.timestamp).total_seconds()
        if delta <= 1e-9:
            return (0.0, 0.0)
        refresh_hz = 1.0 / delta
        stale_ms = delta * 1000.0
        return (refresh_hz, stale_ms)

    def _depth_shape_features(
        self,
        prices: np.ndarray,
        sizes: np.ndarray,
        tick_size: float,
        *,
        side: str,
    ) -> tuple[float, float]:
        if prices.size == 0 or sizes.size == 0 or tick_size <= 0.0:
            return (0.0, 0.0)

        if side == "bid":
            offsets = (prices[0] - prices) / tick_size
        else:
            offsets = (prices - prices[0]) / tick_size

        offsets = np.clip(offsets, 0.0, None)
        if np.allclose(offsets, offsets[0]):
            return (0.0, 0.0)

        best_size = float(max(sizes[0], 1e-9))
        norm_sizes = np.clip(sizes / best_size, 0.0, None)

        degree = 2 if offsets.size >= 3 else 1
        try:
            coeffs = np.polyfit(offsets, norm_sizes, deg=degree)
        except np.linalg.LinAlgError:
            return (0.0, 0.0)

        if degree == 1:
            slope = float(coeffs[0])
            curvature = 0.0
        else:
            a2, a1, _ = coeffs
            slope = float(a1)
            curvature = float(2.0 * a2)

        return (slope, curvature)

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
    def _liquidity_ratio(
        best_bid_size: float,
        best_ask_size: float,
        spread: float,
        tick_size: float,
    ) -> float:
        numerator = max(best_bid_size + best_ask_size, 0.0)
        denominator = max(spread, tick_size, 1e-9)
        ratio = numerator / denominator
        if not isfinite(ratio) or ratio < 0.0:
            return 0.0
        # Clamp to avoid pathological spikes when spreads collapse.
        return float(min(ratio, 1e9))

    @staticmethod
    def _select_first_existing(columns: Iterable[str], candidates: set[str]) -> list[str]:
        available = [col for col in columns if col in candidates]
        return available
