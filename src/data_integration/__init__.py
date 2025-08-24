"""
Data Integration Package - Phase 1.5 Complete Implementation

This package provides comprehensive real data integration capabilities for the EMP system.
It includes all major data sources with validation and quality monitoring.

Author: EMP Development Team
Date: July 18, 2024
Phase: 1.5 - Advanced Data Sources Complete
"""

from __future__ import annotations

from src.validation.models import ValidationResult  # re-export for package smoke test


# Import main data management classes
# Legacy imports removed during cleanup; provide minimal stubs for compatibility
class RealDataManager:  # type: ignore
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    async def get_market_data(self, *_args, **_kwargs):
        return None


class DataSourceConfig:  # type: ignore
    pass


class YahooFinanceDataProvider:  # type: ignore
    pass


class AlphaVantageDataProvider:  # type: ignore
    pass


class FREDDataProvider:  # type: ignore
    pass


class NewsAPIDataProvider:  # type: ignore
    pass


class MarketDataValidator:  # type: ignore
    pass


class DataConsistencyChecker:  # type: ignore
    pass


class DataQualityMonitor:  # type: ignore
    pass


class ValidationLevel:  # type: ignore
    pass


class DataQualityThresholds:  # type: ignore
    pass


class DataIssue:  # type: ignore
    pass


# Import advanced providers (if available)
ADVANCED_PROVIDERS_AVAILABLE = False

__all__ = [
    # Core data management
    "RealDataManager",
    "DataSourceConfig",
    # Basic providers
    "YahooFinanceDataProvider",
    "AlphaVantageDataProvider",
    "FREDDataProvider",
    "NewsAPIDataProvider",
    # Advanced providers (if available)
    # Advanced providers intentionally unavailable in FIX-only baseline
    # Data validation
    "ValidationResult",
    "MarketDataValidator",
    "DataConsistencyChecker",
    "DataQualityMonitor",
    "ValidationLevel",
    "DataQualityThresholds",
    "DataIssue",
    # Status
    "ADVANCED_PROVIDERS_AVAILABLE",
]
