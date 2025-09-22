from __future__ import annotations

from typing import Any

class InstitutionalFootprint:
    smart_money_flow: float
    institutional_bias: str
    confidence_score: float

class InstitutionalFootprintHunter:
    def __init__(self) -> None: ...
    async def analyze_institutional_footprint(
        self, market_data: Any, symbol: str = ...
    ) -> InstitutionalFootprint: ...
