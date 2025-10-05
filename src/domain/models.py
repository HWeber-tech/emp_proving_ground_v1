"""
EMP Domain Models v1.1

Shared domain models used across all layers.
Separates domain concerns from infrastructure concerns.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel


class ExecutionReport(BaseModel):
    """Execution report for trade confirmations"""

    event_id: str
    timestamp: datetime
    source: str
    trade_intent_id: str
    action: str
    status: str
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    price: float
    order_id: str

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }


__all__ = ["ExecutionReport"]
