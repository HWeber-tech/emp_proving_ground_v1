from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass
class RiskConfig:
    max_risk_per_trade_pct: Decimal = Decimal("0.02")
    max_drawdown_pct: Decimal = Decimal("0.25")


class RiskManager:
    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def validate_trade(self, size: Decimal, entry_price: Decimal) -> bool:
        return size > 0 and entry_price > 0


