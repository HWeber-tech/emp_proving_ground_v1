"""Legacy Yahoo Finance organ shim (retired).

This module previously exposed the YahooFinanceOrgan sensory adapter. The
canonical market data implementation now lives under
``src.data_foundation.ingest.yahoo_gateway`` as ``YahooMarketDataGateway``.

Importers must migrate to the canonical gateway. The legacy organ shim raises a
ModuleNotFoundError with guidance to prevent silent fallbacks.
"""

from __future__ import annotations

raise ModuleNotFoundError(
    "src.sensory.organs.yahoo_finance_organ was removed. Use "
    "src.data_foundation.ingest.yahoo_gateway.YahooMarketDataGateway instead."
)
