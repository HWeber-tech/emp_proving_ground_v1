"""
FRED-like calendar ingestion for WHY dimension using public APIs where available.
This is a placeholder; customize with your preferred source.
"""

from __future__ import annotations


from src.data_foundation.schemas import MacroEvent


def fetch_fred_calendar(start: str, end: str) -> list[MacroEvent]:
    # Placeholder: return empty to keep pipeline stable if no creds/API
    return []
