"""
Data Integration Package - Phase 1.5 Complete Implementation

This package provides comprehensive real data integration capabilities for the EMP system.
It includes all major data sources with validation and quality monitoring.

Author: EMP Development Team
Date: July 18, 2024
Phase: 1.5 - Advanced Data Sources Complete
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

# Import advanced providers (if available)
try:
    from .alpha_vantage_integration import (
        AlphaVantageProvider,
        AlphaVantageConfig
    )
    from .fred_integration import (
        FREDProvider,
        FREDConfig
    )
    from .newsapi_integration import (
        NewsAPIProvider,
        NewsAPIConfig
    )
    ADVANCED_PROVIDERS_AVAILABLE = True
except ImportError:
    ADVANCED_PROVIDERS_AVAILABLE = False

__all__ = [
    # Core data management
    'RealDataManager',
    'DataSourceConfig',
    
    # Basic providers
    'YahooFinanceDataProvider',
    'AlphaVantageDataProvider',
    'FREDDataProvider',
    'NewsAPIDataProvider',
    
    # Advanced providers (if available)
    'AlphaVantageProvider',
    'AlphaVantageConfig',
    'FREDProvider',
    'FREDConfig',
    'NewsAPIProvider',
    'NewsAPIConfig',
    
    # Data validation
    'MarketDataValidator',
    'DataConsistencyChecker',
    'DataQualityMonitor',
    'ValidationLevel',
    'ValidationResult',
    'DataQualityThresholds',
    'DataIssue',
    
    # Status
    'ADVANCED_PROVIDERS_AVAILABLE'
] 
