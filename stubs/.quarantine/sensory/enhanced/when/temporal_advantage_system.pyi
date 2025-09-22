from __future__ import annotations

from datetime import datetime
from typing import Any, Tuple

class TemporalAdvantage:
    session_transition_score: float
    economic_calendar_impact: dict[str, float]
    microstructure_timing: dict[str, Any]
    volatility_regime: str
    optimal_entry_window: Tuple[datetime, datetime]
    confidence_score: float

class TemporalAdvantageSystem:
    def __init__(self) -> None: ...
    async def analyze_timing(self, market_data: Any) -> TemporalAdvantage: ...
