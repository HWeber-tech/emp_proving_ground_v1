from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.strategy.engine import BaseStrategy


class MovingAverageStrategy(BaseStrategy):
    def __init__(self, strategy_id: str, symbols: List[str], params: Dict[str, Any]):
        super().__init__(strategy_id, symbols)
        self.fast = int(params.get("fast_period", 20))
        self.slow = int(params.get("slow_period", 50))

    async def generate_signal(self, market_data: List[Any], symbol: str) -> Optional[Dict[str, Any]]:
        return None


