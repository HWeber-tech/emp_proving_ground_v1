from __future__ import annotations
from typing import List, Union, SupportsInt, SupportsIndex, cast
from collections.abc import Mapping
from src.core.strategy.engine import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    def __init__(self, strategy_id: str, symbols: List[str], params: Mapping[str, object]):
        super().__init__(strategy_id, symbols)
        val_period = cast("Union[str, SupportsInt, SupportsIndex]", params.get("period", 20))
        self.period = int(val_period)

    async def generate_signal(self, market_data: object, symbol: str) -> object:
        return None


