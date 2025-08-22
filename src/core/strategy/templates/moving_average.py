from __future__ import annotations
from typing import List, Union, SupportsInt, SupportsIndex, cast
from collections.abc import Mapping

from src.core.strategy.engine import BaseStrategy


class MovingAverageStrategy(BaseStrategy):
    def __init__(self, strategy_id: str, symbols: List[str], params: Mapping[str, object]):
        super().__init__(strategy_id, symbols)
        val_fast = cast("Union[str, SupportsInt, SupportsIndex]", params.get("fast_period", 20))
        self.fast = int(val_fast)
        val_slow = cast("Union[str, SupportsInt, SupportsIndex]", params.get("slow_period", 50))
        self.slow = int(val_slow)

    async def generate_signal(self, market_data: object, symbol: str) -> object:
        return None


