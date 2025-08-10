from __future__ import annotations

from typing import Dict, List, Any, Optional

from src.core.strategy.engine import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    def __init__(self, strategy_id: str, symbols: List[str], params: Dict[str, Any]):
        super().__init__(strategy_id, symbols)
        self.period = int(params.get("period", 20))

    async def generate_signal(self, market_data: List[Any], symbol: str) -> Optional[Dict[str, Any]]:
        return None


