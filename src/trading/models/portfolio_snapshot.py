from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from .position import Position


@dataclass
class PortfolioSnapshot:
    """Portfolio snapshot (package-local definition for typing/runtime use)."""
    total_value: float
    cash_balance: float
    positions: List[Position]
    unrealized_pnl: float
    realized_pnl: float
    timestamp: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now()

    @property
    def position_value(self) -> float:
        """Total value of all positions"""
        return sum(pos.value for pos in self.positions)

    @property
    def total_pnl(self) -> float:
        """Total P&L"""
        return self.unrealized_pnl + self.realized_pnl