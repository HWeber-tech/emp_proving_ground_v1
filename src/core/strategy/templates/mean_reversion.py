from __future__ import annotations

from collections.abc import Mapping
from typing import SupportsIndex, SupportsInt, Union, cast

from src.core.strategy.engine import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    def __init__(self, strategy_id: str, symbols: list[str], params: Mapping[str, object]) -> None:
        super().__init__(strategy_id, symbols)
        val_period = cast("Union[str, SupportsInt, SupportsIndex]", params.get("period", 20))
        self.period = int(val_period)

    async def generate_signal(self, market_data: object, symbol: str) -> object:
        return None
