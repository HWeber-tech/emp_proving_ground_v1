"""
Microstructure features for WHAT sense.

All analysis remains in sensory layer. Inputs are generic lists to avoid
coupling to operational types.
"""

from __future__ import annotations

from typing import List, Tuple, Dict
from collections import deque


def _best(book_side: List[Tuple[float, float]], reverse: bool) -> Tuple[float, float]:
    if not book_side:
        return 0.0, 0.0
    # book_side expected unsorted prices; determine best by price
    best = max(book_side, key=lambda x: x[0]) if reverse else min(book_side, key=lambda x: x[0])
    return float(best[0]), float(best[1])


def compute_features(
    bids: List[Tuple[float, float]],
    asks: List[Tuple[float, float]],
    levels: int = 5,
) -> Dict[str, float]:
    """Compute microstructure features from L2 book.

    Args:
      bids: list of (price, size), any order
      asks: list of (price, size), any order
      levels: depth levels to aggregate for imbalance/depth
    Returns: dict of features (spread, mid, microprice, imbalances, depths)
    """
    # Sort
    bids_sorted = sorted([(p, s) for p, s in bids if s > 0], key=lambda x: x[0], reverse=True)
    asks_sorted = sorted([(p, s) for p, s in asks if s > 0], key=lambda x: x[0])

    best_bid, bid_size = _best(bids_sorted, reverse=True)
    best_ask, ask_size = _best(asks_sorted, reverse=False)

    spread = (best_ask - best_bid) if (best_ask > 0 and best_bid > 0) else 0.0
    mid = (best_ask + best_bid) / 2.0 if (best_ask > 0 and best_bid > 0) else 0.0

    # Microprice (size-weighted mid)
    denom = bid_size + ask_size
    microprice = ((best_ask * bid_size + best_bid * ask_size) / denom) if denom > 0 else mid

    # Depth aggregates up to levels
    bid_depth = sum(s for _, s in bids_sorted[:levels])
    ask_depth = sum(s for _, s in asks_sorted[:levels])

    # Imbalance (top level and aggregate)
    top_imbalance = (bid_size - ask_size) / (bid_size + ask_size) if (bid_size + ask_size) > 0 else 0.0
    depth_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0.0

    return {
        "spread": float(spread),
        "mid": float(mid),
        "microprice": float(microprice),
        "bid_depth_l{}".format(levels): float(bid_depth),
        "ask_depth_l{}".format(levels): float(ask_depth),
        "top_imbalance": float(top_imbalance),
        "depth_imbalance_l{}".format(levels): float(depth_imbalance),
    }


def compute_liquidity_pockets(
    bids: List[Tuple[float, float]],
    asks: List[Tuple[float, float]],
    levels: int = 10,
    pocket_factor: float = 1.5,
) -> Dict[str, float]:
    """Detect liquidity pockets as sizes exceeding pocket_factor * mean(size) in top N levels.

    Returns counts and maximum pocket strength (ratio to mean) for each side.
    """
    def side_stats(side):
        s = sorted([(p, s) for p, s in side if s > 0], key=lambda x: x[0], reverse=(side is bids))[:levels]
        sizes = [sz for _, sz in s]
        if not sizes:
            return 0, 0.0
        mean_sz = sum(sizes) / len(sizes)
        if mean_sz <= 0:
            return 0, 0.0
        strengths = [sz / mean_sz for sz in sizes if sz / mean_sz >= pocket_factor]
        return len(strengths), (max(strengths) if strengths else 0.0)

    bid_count, bid_strength = side_stats(bids)
    ask_count, ask_strength = side_stats(asks)
    return {
        "bid_pocket_count_l{}".format(levels): float(bid_count),
        "ask_pocket_count_l{}".format(levels): float(ask_count),
        "bid_pocket_strength": float(bid_strength),
        "ask_pocket_strength": float(ask_strength),
    }


def compute_volatility_seeds(mid_prices: List[float]) -> Dict[str, float]:
    """Compute short-horizon volatility seeds from mid price series.

    Returns rolling std devs over 5/10/20 samples and mean absolute change.
    """
    if not mid_prices:
        return {"std5": 0.0, "std10": 0.0, "std20": 0.0, "mean_abs_change": 0.0}
    def rolling_std(window: int) -> float:
        if len(mid_prices) < 2:
            return 0.0
        arr = mid_prices[-window:]
        n = len(arr)
        mean = sum(arr) / n
        var = sum((x - mean) ** 2 for x in arr) / max(1, n - 1)
        return var ** 0.5
    def mean_abs_change() -> float:
        if len(mid_prices) < 2:
            return 0.0
        diffs = [abs(b - a) for a, b in zip(mid_prices[:-1], mid_prices[1:])]
        return sum(diffs) / len(diffs)
    return {
        "std5": float(rolling_std(5)),
        "std10": float(rolling_std(10)),
        "std20": float(rolling_std(20)),
        "mean_abs_change": float(mean_abs_change()),
    }


class RollingMicrostructure:
    """Maintains rolling windows over mid price and OBI for offline replay analysis."""

    def __init__(self, window: int = 20):
        self.window = window
        self.mids: deque = deque(maxlen=window)
        self.obi: deque = deque(maxlen=window)

    def update(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> Dict[str, float]:
        f = compute_features(bids, asks)
        self.mids.append(f["mid"]) if f["mid"] else None
        # Order book imbalance (top)
        self.obi.append(f.get("top_imbalance", 0.0))
        # Volatility seeds from mids
        vol = compute_volatility_seeds(list(self.mids))
        # Mean OBI
        mean_obi = sum(self.obi) / len(self.obi) if self.obi else 0.0
        # Simple mid reversion seed: mid - mean(mid)
        rev = 0.0
        if self.mids:
            m = list(self.mids)
            rev = (m[-1] - (sum(m) / len(m)))
        out = {**f, **vol, "mean_obi": float(mean_obi), "mid_reversion": float(rev)}
        return out


