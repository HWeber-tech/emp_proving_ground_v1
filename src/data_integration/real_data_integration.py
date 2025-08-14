"""
Stub module to satisfy optional import: src.data_integration.real_data_integration
Runtime-safe no-op implementation for validation flows.
"""

from __future__ import annotations

from typing import Any, Optional


class RealDataManager:
    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        self.config = config or {}

    async def get_market_data(self, symbol: str, source: str | None = None) -> Optional[Any]:
        """
        No-op stub: returns None to indicate unavailable data.
        Validation code treats None as 'no data available' without failing at import time.
        """
        return None