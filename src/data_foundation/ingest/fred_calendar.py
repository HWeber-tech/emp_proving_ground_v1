"""
FRED-like calendar ingestion for WHY dimension using public APIs where available.
This is a placeholder; customize with your preferred source.
"""

from __future__ import annotations

from typing import List

from src.data_foundation.schemas import MacroEvent


def fetch_fred_calendar(start: str, end: str) -> List[MacroEvent]:
    # Placeholder: return empty to keep pipeline stable if no creds/API
    return []


