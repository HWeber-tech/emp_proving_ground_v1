from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class StrategyPerformance:
    total_return: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0


class BaseStrategy:
    def __init__(self, strategy_id: str, symbols: List[str]):
        self.strategy_id = strategy_id
        self.symbols = symbols
        self.performance = StrategyPerformance()

    async def generate_signal(self, market_data, symbol: str):
        raise NotImplementedError


class StrategyEngine:
    """Consolidated, minimal strategy engine surface."""

    def __init__(self) -> None:
        self._strategies: Dict[str, BaseStrategy] = {}
        self._performance: Dict[str, StrategyPerformance] = {}
        self._last_updated = datetime.utcnow()

    def register(self, strategy: BaseStrategy) -> bool:
        sid = strategy.strategy_id
        if not sid or sid in self._strategies:
            return False
        self._strategies[sid] = strategy
        self._performance[sid] = strategy.performance
        return True

    def unregister(self, strategy_id: str) -> bool:
        if strategy_id not in self._strategies:
            return False
        del self._strategies[strategy_id]
        del self._performance[strategy_id]
        return True

    def performance(self, strategy_id: str) -> Optional[StrategyPerformance]:
        return self._performance.get(strategy_id)


