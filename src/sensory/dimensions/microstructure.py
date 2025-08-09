"""
Microstructure features for WHAT sense.

All analysis remains in sensory layer. Inputs are generic lists to avoid
coupling to operational types.
"""

from __future__ import annotations

from typing import List, Tuple, Dict


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


