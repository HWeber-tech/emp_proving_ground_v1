from __future__ import annotations

import importlib
from decimal import Decimal
from typing import Any


def _risk_cfg(sym: str) -> Any:
    return getattr(importlib.import_module("src.config.risk.risk_config"), sym)


class RiskManager:
    def __init__(self, config: Any | None = None) -> None:
        RiskConfig = _risk_cfg("RiskConfig")
        self.config = config or RiskConfig()

    def validate_trade(self, size: Decimal, entry_price: Decimal) -> bool:
        return size > 0 and entry_price > 0
