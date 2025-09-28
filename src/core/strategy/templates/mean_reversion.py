from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import fmean, pstdev
from typing import Any, SupportsIndex, SupportsInt, Union, cast

from src.core.strategy.engine import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """Mean-reversion template that emits Bollinger-band style signals.

    The roadmap calls for a production-ready mean-reversion playbook with
    configurable bands.  This implementation keeps the surface intentionally
    light-weight so it can operate during paper-trading without introducing
    heavy numerical dependencies.  The strategy expects a history of closing
    prices per symbol and returns a signal payload describing the current
    band positioning.
    """

    def __init__(self, strategy_id: str, symbols: list[str], params: Mapping[str, object]) -> None:
        super().__init__(strategy_id, symbols)
        val_period = cast(Union[str, SupportsInt, SupportsIndex], params.get("period", 20))
        try:
            period = int(val_period)
        except (TypeError, ValueError):
            period = 20
        self.period = max(1, period)

        num_std_raw = params.get("num_std", 2.0)
        try:
            self.num_std = float(cast(Union[str, float], num_std_raw))
        except (TypeError, ValueError):
            self.num_std = 2.0
        if self.num_std <= 0:
            self.num_std = 2.0

    async def generate_signal(self, market_data: object, symbol: str) -> dict[str, Any]:
        """Return the mean-reversion signal for *symbol*.

        The strategy accepts multiple market-data shapes:
        - Mapping[symbol -> Sequence[float|dict]]
        - Sequence of floats or dicts containing ``close``/``price`` values
        - Objects exposing ``close`` attributes (e.g. MarketData instances)
        """

        prices = self._extract_price_history(market_data, symbol)
        observations = len(prices)
        if observations < self.period:
            return {
                "symbol": symbol,
                "signal": "HOLD",
                "reason": "insufficient_history",
                "observations": observations,
                "required_history": self.period,
            }

        window = prices[-self.period :]
        moving_average = fmean(window)
        std_dev = pstdev(window) if self.period > 1 else 0.0
        price = window[-1]
        upper_band = moving_average + self.num_std * std_dev
        lower_band = moving_average - self.num_std * std_dev

        if std_dev <= 1e-9:
            signal = "HOLD"
            reason = "flat_volatility"
            z_score = 0.0
        else:
            z_score = (price - moving_average) / std_dev
            if price <= lower_band:
                signal = "BUY"
                reason = "price_below_lower_band"
            elif price >= upper_band:
                signal = "SELL"
                reason = "price_above_upper_band"
            else:
                signal = "HOLD"
                reason = "price_within_bands"

        return {
            "symbol": symbol,
            "signal": signal,
            "reason": reason,
            "price": price,
            "moving_average": moving_average,
            "upper_band": upper_band,
            "lower_band": lower_band,
            "bandwidth": upper_band - lower_band,
            "standard_deviation": std_dev,
            "z_score": z_score,
            "observations": observations,
        }

    # ------------------------------------------------------------------
    def _extract_price_history(self, market_data: object, symbol: str) -> list[float]:
        if isinstance(market_data, Mapping):
            if symbol in market_data:
                raw = market_data[symbol]
            else:
                raw = (
                    market_data.get("prices")
                    or market_data.get("close")
                    or market_data.get("data")
                    or market_data
                )
        else:
            raw = market_data

        if hasattr(raw, "to_list") and callable(getattr(raw, "to_list")):
            raw = raw.to_list()  # type: ignore[assignment]
        elif hasattr(raw, "tolist") and callable(getattr(raw, "tolist")):
            raw = raw.tolist()  # type: ignore[assignment]

        if isinstance(raw, Mapping):
            for key in ("close", "price", "mid", "last"):
                if key in raw:
                    raw = raw[key]
                    break

        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            return [price for price in map(self._coerce_price, raw) if price is not None]

        price = self._coerce_price(raw)
        return [price] if price is not None else []

    def _coerce_price(self, item: object) -> float | None:
        if item is None:
            return None
        if isinstance(item, (int, float)):
            return float(item)
        if isinstance(item, str):
            try:
                return float(item)
            except ValueError:
                return None
        if isinstance(item, Mapping):
            for key in ("close", "price", "mid", "last"):
                if key in item:
                    return self._coerce_price(item[key])
            return None
        if hasattr(item, "close"):
            return self._coerce_price(getattr(item, "close"))
        if hasattr(item, "__float__"):
            try:
                return float(item)  # type: ignore[misc]
            except Exception:
                return None
        return None
