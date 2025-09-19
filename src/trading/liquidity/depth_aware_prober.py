from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from threading import RLock
from typing import Deque, Dict, Iterable, Mapping

from src.core.base import MarketData

__all__ = ["DepthAwareLiquidityProber"]


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


@dataclass
class _DepthSnapshot:
    price: float
    spread: float
    depth: float
    imbalance: float
    volume: float
    data_quality: float
    timestamp: datetime


class DepthAwareLiquidityProber:
    """Estimate tradeable liquidity using recent market depth observations."""

    def __init__(
        self,
        *,
        max_history: int = 64,
        decay: float = 0.65,
        min_quality: float = 0.2,
    ) -> None:
        if max_history <= 0:
            raise ValueError("max_history must be positive")
        if not (0.0 < decay <= 1.0):
            raise ValueError("decay must fall in (0, 1]")
        self.max_history = int(max_history)
        self.decay = float(decay)
        self.min_quality = float(min_quality)
        self._history: Dict[str, Deque[_DepthSnapshot]] = {}
        self._lock = RLock()
        self._last_probe_meta: dict[str, float | int | str] | None = None

    def record_snapshot(self, symbol: str, market_data: MarketData) -> None:
        """Store a lightweight depth snapshot for later liquidity estimation."""

        price = _to_float(getattr(market_data, "mid_price", None), default=market_data.mid_price)
        spread = _to_float(getattr(market_data, "spread", None), default=market_data.spread)
        depth = _to_float(getattr(market_data, "depth", None), default=0.0)
        volume = _to_float(getattr(market_data, "volume", None), default=0.0)
        imbalance = _to_float(getattr(market_data, "order_imbalance", None), default=0.0)
        quality = _to_float(getattr(market_data, "data_quality", None), default=1.0)

        if depth <= 0.0:
            # Fall back to a volume-derived estimate if explicit depth is missing
            depth = max(volume * 0.2, 0.0)
        if spread <= 0.0:
            spread = max(price * 0.0001, 1e-6)

        snapshot = _DepthSnapshot(
            price=price if price else _to_float(getattr(market_data, "close", None), default=0.0),
            spread=spread,
            depth=depth,
            imbalance=max(-1.0, min(1.0, imbalance)),
            volume=volume,
            data_quality=max(self.min_quality, min(1.0, quality if quality else 0.0)),
            timestamp=getattr(market_data, "timestamp", datetime.utcnow()),
        )

        with self._lock:
            history = self._history.setdefault(symbol, deque(maxlen=self.max_history))
            history.append(snapshot)

    async def probe_liquidity(
        self, symbol: str, price_levels: Iterable[float], side: str
    ) -> Mapping[float, float]:
        """Estimate available liquidity at the requested price levels."""

        levels = [float(level) for level in price_levels]
        if not levels:
            return {}

        side_key = side.lower()

        with self._lock:
            history = list(self._history.get(symbol, ()))

        results: dict[float, float] = {level: 0.0 for level in levels}
        if not history:
            self._last_probe_meta = {
                "symbol": symbol,
                "observations_used": 0,
                "average_quality": 0.0,
                "average_depth": 0.0,
                "average_spread": 0.0,
            }
            return results

        total_quality = 0.0
        total_depth = 0.0
        total_spread = 0.0

        for idx, snapshot in enumerate(reversed(history)):
            weight = self.decay**idx
            total_quality += snapshot.data_quality
            total_depth += snapshot.depth
            total_spread += snapshot.spread
            for level in levels:
                results[level] += weight * self._estimate_volume(snapshot, level, side_key)

        count = len(history)
        self._last_probe_meta = {
            "symbol": symbol,
            "observations_used": count,
            "average_quality": total_quality / count if count else 0.0,
            "average_depth": total_depth / count if count else 0.0,
            "average_spread": total_spread / count if count else 0.0,
        }
        return results

    def calculate_liquidity_confidence_score(
        self, probe_results: Mapping[float, float], intended_volume: float
    ) -> float:
        if not probe_results:
            return 0.0

        total_liquidity = sum(max(0.0, _to_float(v, default=0.0)) for v in probe_results.values())
        if intended_volume <= 0:
            return 1.0

        coverage = max(0.0, total_liquidity / max(intended_volume, 1e-9))
        score = 1.0 - math.exp(-coverage)

        meta = self._last_probe_meta or {}
        quality = _to_float(meta.get("average_quality"), default=1.0)
        quality_modifier = 0.7 + 0.3 * max(self.min_quality, min(1.0, quality))
        score *= quality_modifier

        return max(0.0, min(1.0, score))

    def get_probe_summary(self, probe_results: Mapping[float, float]) -> Mapping[str, float | int | str | None]:
        total = sum(_to_float(v, default=0.0) for v in probe_results.values())
        levels = list(probe_results.keys())
        peak_level = None
        peak_liquidity = 0.0
        if levels:
            peak_level = max(levels, key=lambda lvl: _to_float(probe_results[lvl], default=0.0))
            peak_liquidity = _to_float(probe_results[peak_level], default=0.0)

        summary: dict[str, float | int | str | None] = {
            "evaluated_levels": len(levels),
            "total_liquidity": total,
            "peak_level": peak_level,
            "peak_liquidity": peak_liquidity,
            "average_liquidity": (total / len(levels)) if levels else 0.0,
        }
        if self._last_probe_meta:
            summary.update(self._last_probe_meta)
        return summary

    def _estimate_volume(self, snapshot: _DepthSnapshot, level: float, side_key: str) -> float:
        price_distance = abs(level - snapshot.price)
        spread_band = max(snapshot.spread * 6.0, 1e-6)
        distance_factor = max(0.0, 1.0 - price_distance / spread_band)

        if side_key.startswith("buy"):
            bias = snapshot.imbalance
        elif side_key.startswith("sell"):
            bias = -snapshot.imbalance
        else:
            bias = 0.0
        imbalance_factor = max(0.3, 1.0 + 0.5 * bias)

        quality_factor = max(self.min_quality, min(1.0, snapshot.data_quality))
        base_depth = max(snapshot.depth, snapshot.volume * 0.1)

        estimate = base_depth * distance_factor * imbalance_factor * quality_factor
        return max(0.0, estimate)
