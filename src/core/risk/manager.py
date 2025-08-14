from __future__ import annotations

from decimal import Decimal

from src.config.risk.risk_config import RiskConfig


class RiskManager:
    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def validate_trade(self, size: Decimal, entry_price: Decimal) -> bool:
        return size > 0 and entry_price > 0

