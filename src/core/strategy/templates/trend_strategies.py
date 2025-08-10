from __future__ import annotations

from typing import Dict, List, Optional, Any
from datetime import datetime

from src.core.strategy.engine import BaseStrategy, StrategyPerformance


class TrendFollowing(BaseStrategy):
    def __init__(self, strategy_id: str, symbols: List[str], params: Dict[str, Any]):
        super().__init__(strategy_id, symbols)
        self.params = params

    async def generate_signal(self, market_data, symbol: str):
        return None

