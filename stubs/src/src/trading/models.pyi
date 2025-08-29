from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Any

@dataclass
class PortfolioSnapshot:
    total_value: float
    cash_balance: float
    positions: List[Any]
    unrealized_pnl: float
    realized_pnl: float
    timestamp: Optional[datetime] = None

__all__ = ["PortfolioSnapshot"]