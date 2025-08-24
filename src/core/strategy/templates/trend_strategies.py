from __future__ import annotations

from src.core.strategy.engine import BaseStrategy


class TrendFollowing(BaseStrategy):
    def __init__(self, strategy_id: str, symbols: list[str], params: dict[str, object]) -> None:
        super().__init__(strategy_id, symbols)
        self.params = params

    async def generate_signal(self, market_data: object, symbol: str) -> object:
        return None
