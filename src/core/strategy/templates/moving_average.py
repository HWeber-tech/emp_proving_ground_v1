from __future__ import annotations

from collections.abc import Mapping
from typing import SupportsIndex, SupportsInt, Union, cast

from src.core.strategy.engine import BaseStrategy


class MovingAverageStrategy(BaseStrategy):
    def __init__(self, strategy_id: str, symbols: list[str], params: Mapping[str, object]) -> None:
        super().__init__(strategy_id, symbols)
        val_fast = cast("Union[str, SupportsInt, SupportsIndex]", params.get("fast_period", 20))
        self.fast = int(val_fast)
        val_slow = cast("Union[str, SupportsInt, SupportsIndex]", params.get("slow_period", 50))
        self.slow = int(val_slow)

    async def generate_signal(self, market_data: object, symbol: str) -> object:
        return None
