"""
EMP Domain Models v1.1

Shared domain models used across all layers.
Separates domain concerns from infrastructure concerns.
"""

import logging
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# RiskConfig is unified under src.core.risk.manager
try:
    from src.core.risk.manager import RiskConfig  # type: ignore
except Exception:
    RiskConfig = object  # type: ignore


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
            Decimal: str
        }




# Re-export canonical implementations instead of defining duplicates
from importlib import import_module as _imp  # type: ignore

try:
    _core_mod = _imp("src.core")  # prefer canonical implementations in src/core.py
    InstrumentProvider = getattr(_core_mod, "InstrumentProvider")  # type: ignore
except Exception:
    class InstrumentProvider:  # type: ignore
        pass

try:
    _core_mod = _imp("src.core")
    CurrencyConverter = getattr(_core_mod, "CurrencyConverter")  # type: ignore
except Exception:
    class CurrencyConverter:  # type: ignore
        pass
