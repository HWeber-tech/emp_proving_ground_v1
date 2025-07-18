"""
Data Package - Phase 1 Real Data Integration

This package provides real data integration capabilities for the EMP system.
It includes data providers, validation, and quality monitoring.

Author: EMP Development Team
Date: July 18, 2024
Phase: 1 - Real Data Foundation
"""

# Import main data management classes
from .real_data_integration import (
    RealDataManager,
    DataSourceConfig,
    YahooFinanceDataProvider,
    AlphaVantageDataProvider,
    FREDDataProvider,
    NewsAPIDataProvider
)

from .data_validation import (
    MarketDataValidator,
    DataConsistencyChecker,
    DataQualityMonitor,
    ValidationLevel,
    ValidationResult,
    DataQualityThresholds,
    DataIssue
)

__all__ = [
    'RealDataManager',
    'DataSourceConfig',
    'YahooFinanceDataProvider',
    'AlphaVantageDataProvider',
    'FREDDataProvider',
    'NewsAPIDataProvider',
    'MarketDataValidator',
    'DataConsistencyChecker',
    'DataQualityMonitor',
    'ValidationLevel',
    'ValidationResult',
    'DataQualityThresholds',
    'DataIssue'
] 