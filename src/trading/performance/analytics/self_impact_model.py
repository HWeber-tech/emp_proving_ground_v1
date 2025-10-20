"""Self-impact model quantifying how EMP trades perturb the local order book.

The self-impact model tracks how executed trades reshape immediate order book
conditions and short-horizon realised volatility.  It focuses on intuitive
microstructure deltas (mid-price dislocation, spread change, depth consumed,
and liquidity imbalance shift) along with volatility drift so that execution
policy can reason about the *footprint* each trade leaves behind.

The implementation intentionally keeps dependencies light.  It operates purely
on ``OrderBookSnapshot`` objects and price sequences, making it suitable for
unit tests and offline analytics pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

import numpy as np

from src.trading.models.trade import Trade
from src.trading.order_management.order_book.snapshot import (
    OrderBookLevel,
    OrderBookSnapshot,
)


@dataclass(frozen=True)
class SelfImpactMetrics:
    """Structured record describing the impact of a single trade."""

    trade_id: str
    symbol: str
    timestamp: datetime
    price_impact_bps: float
    spread_change_bps: float
    depth_consumed: float
    imbalance_change: float
    volatility_before: float
    volatility_after: float
    volatility_change: float
    metadata: Mapping[str, float]


class SelfImpactModel:
    """Calculate market footprint metrics for EMP's executed trades."""

    def __init__(self, *, volatility_floor: float = 1e-9) -> None:
        if volatility_floor <= 0.0:
            raise ValueError("volatility_floor must be positive")
        self._volatility_floor = float(volatility_floor)

    # ------------------------------------------------------------------
    def evaluate_trade(
        self,
        trade: Trade,
        *,
        pre_trade_snapshot: OrderBookSnapshot,
        post_trade_snapshot: OrderBookSnapshot,
        pre_trade_mid_prices: Sequence[float] | None = None,
        post_trade_mid_prices: Sequence[float] | None = None,
    ) -> SelfImpactMetrics:
        """Quantify how the trade perturbed local order book state."""

        if trade.symbol != pre_trade_snapshot.symbol:
            raise ValueError("trade symbol mismatch with pre-trade snapshot")
        if trade.symbol != post_trade_snapshot.symbol:
            raise ValueError("trade symbol mismatch with post-trade snapshot")

        mid_before = float(pre_trade_snapshot.mid_price)
        mid_after = float(post_trade_snapshot.mid_price)
        spread_before = float(pre_trade_snapshot.spread)
        spread_after = float(post_trade_snapshot.spread)

        divisor_mid = mid_before if mid_before > 0.0 else mid_after if mid_after > 0.0 else 1.0
        price_impact_bps = ((mid_after - mid_before) / divisor_mid) * 10000.0

        spread_denominator = divisor_mid
        spread_change_bps = ((spread_after - spread_before) / spread_denominator) * 10000.0

        bid_before = self._total_volume(pre_trade_snapshot.bids)
        ask_before = self._total_volume(pre_trade_snapshot.asks)
        bid_after = self._total_volume(post_trade_snapshot.bids)
        ask_after = self._total_volume(post_trade_snapshot.asks)

        depth_consumed = self._depth_consumed(trade.side, ask_before, ask_after, bid_before, bid_after)

        imbalance_before = self._liquidity_imbalance(bid_before, ask_before)
        imbalance_after = self._liquidity_imbalance(bid_after, ask_after)
        imbalance_change = imbalance_after - imbalance_before

        vol_before = self._realised_volatility(pre_trade_mid_prices, fallback=mid_before)
        vol_after = self._realised_volatility(post_trade_mid_prices, fallback=mid_after)
        volatility_change = vol_after - vol_before

        # Express volatility delta as multiplier for downstream guards.
        volatility_multiplier = (
            vol_after / max(vol_before, self._volatility_floor) if vol_after > 0.0 else 0.0
        )

        metadata = {
            "mid_price_before": mid_before,
            "mid_price_after": mid_after,
            "spread_before": spread_before,
            "spread_after": spread_after,
            "bid_volume_before": bid_before,
            "ask_volume_before": ask_before,
            "bid_volume_after": bid_after,
            "ask_volume_after": ask_after,
            "depth_consumed": depth_consumed,
            "imbalance_before": imbalance_before,
            "imbalance_after": imbalance_after,
            "volatility_multiplier": volatility_multiplier,
        }

        return SelfImpactMetrics(
            trade_id=trade.trade_id,
            symbol=trade.symbol,
            timestamp=trade.timestamp,
            price_impact_bps=price_impact_bps,
            spread_change_bps=spread_change_bps,
            depth_consumed=depth_consumed,
            imbalance_change=imbalance_change,
            volatility_before=vol_before,
            volatility_after=vol_after,
            volatility_change=volatility_change,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _total_volume(levels: Sequence[OrderBookLevel]) -> float:
        return float(sum(max(0.0, float(level.volume)) for level in levels))

    @staticmethod
    def _depth_consumed(
        side: str,
        ask_before: float,
        ask_after: float,
        bid_before: float,
        bid_after: float,
    ) -> float:
        side_upper = side.upper()
        if side_upper == "BUY":
            return max(0.0, ask_before - ask_after)
        if side_upper == "SELL":
            return max(0.0, bid_before - bid_after)
        raise ValueError(f"Unsupported trade side '{side}'")

    @staticmethod
    def _liquidity_imbalance(bid_volume: float, ask_volume: float) -> float:
        total = bid_volume + ask_volume
        if total <= 0.0:
            return 0.0
        return (bid_volume - ask_volume) / total

    def _realised_volatility(
        self,
        prices: Sequence[float] | None,
        *,
        fallback: float,
    ) -> float:
        if prices is None:
            prices_array = np.asarray([fallback], dtype=float)
        else:
            prices_array = np.asarray(list(prices), dtype=float)

        if prices_array.size == 0:
            prices_array = np.asarray([fallback], dtype=float)

        finite_mask = np.isfinite(prices_array)
        prices_array = prices_array[finite_mask]
        if prices_array.size < 2:
            return 0.0

        returns = np.diff(prices_array) / prices_array[:-1]
        returns = returns[np.isfinite(returns)]
        if returns.size == 0:
            return 0.0
        ddof = 1 if returns.size > 1 else 0
        return float(np.std(returns, ddof=ddof))


__all__ = ["SelfImpactMetrics", "SelfImpactModel"]
