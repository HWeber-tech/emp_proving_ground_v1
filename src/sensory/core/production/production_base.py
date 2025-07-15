"""
Production Base Classes - Real Data Validation and Core Components

This module provides production-ready base classes that enforce real data usage
and prevent any simulation or mock data from entering the system.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSourceType(Enum):
    """Types of data sources with validation requirements"""
    REAL_API = "real_api"
    REAL_FEED = "real_feed"
    REAL_BROKER = "real_broker"
    SIMULATION = "simulation"  # Explicitly marked for detection
    MOCK = "mock"  # Explicitly marked for detection

class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    RANGING_HIGH_VOLATILITY = "ranging_high_vol"
    RANGING_LOW_VOLATILITY = "ranging_low_vol"
    BREAKOUT_PENDING = "breakout_pending"
    NEWS_DRIVEN = "news_driven"
    ILLIQUID = "illiquid"

class ConfidenceLevel(Enum):
    """Confidence levels for analysis results"""
    VERY_HIGH = "very_high"  # >90%
    HIGH = "high"  # 75-90%
    MEDIUM = "medium"  # 50-75%
    LOW = "low"  # 25-50%
    VERY_LOW = "very_low"  # <25%

@dataclass
class DataValidationResult:
    """Result of data source validation"""
    is_real: bool
    source_type: DataSourceType
    validation_timestamp: datetime
    issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProductionMarketData:
    """
    Production-ready market data with mandatory real data validation
    Replaces the previous MarketData class that allowed simulated data
    """
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    volume: float
    source: str
    data_source_type: DataSourceType
    validation_result: DataValidationResult
    
    # Additional real market data fields
    spread: float = field(init=False)
    mid_price: float = field(init=False)
    
    # Real OHLC data (not estimated)
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    close_price: Optional[float] = None
    
    # Real volume data
    bid_volume: Optional[float] = None
    ask_volume: Optional[float] = None
    
    # Real market microstructure
    tick_direction: Optional[str] = None  # UP, DOWN, UNCHANGED
    trade_count: Optional[int] = None
    
    def __post_init__(self):
        """Validate data and calculate derived fields"""
        # CRITICAL: Validate that data source is real
        if self.data_source_type in [DataSourceType.SIMULATION, DataSourceType.MOCK]:
            raise ValueError(f"SIMULATION/MOCK data not allowed in production: {self.source}")
        
        if not self.validation_result.is_real:
            raise ValueError(f"Non-real data source detected: {self.source}")
        
        # Calculate derived fields
        self.spread = self.ask - self.bid
        self.mid_price = (self.bid + self.ask) / 2.0
        
        # Validate data quality
        self._validate_data_quality()
    
    def _validate_data_quality(self):
        """Validate real market data quality"""
        issues = []
        
        # Basic sanity checks
        if self.bid <= 0 or self.ask <= 0:
            issues.append("Invalid bid/ask prices")
        
        if self.bid >= self.ask:
            issues.append("Bid >= Ask (crossed market)")
        
        if self.spread < 0:
            issues.append("Negative spread")
        
        if self.volume < 0:
            issues.append("Negative volume")
        
        # Check for stale data
        age = datetime.now() - self.timestamp
        if age > timedelta(minutes=5):
            issues.append(f"Stale data: {age.total_seconds():.1f} seconds old")
        
        # Check for unrealistic spreads
        spread_pct = (self.spread / self.mid_price) * 100
        if spread_pct > 1.0:  # >1% spread is suspicious for major pairs
            issues.append(f"Unusually wide spread: {spread_pct:.2f}%")
        
        if issues:
            logger.warning(f"Data quality issues for {self.symbol}: {issues}")
            self.validation_result.issues.extend(issues)

@dataclass
class ProductionDimensionalReading:
    """
    Production-ready dimensional reading with real data enforcement
    """
    dimension: str
    timestamp: datetime
    confidence: ConfidenceLevel
    signal_strength: float  # -1.0 to 1.0
    supporting_evidence: Dict[str, Any]
    data_sources: List[str]
    validation_results: List[DataValidationResult]
    
    # Analysis metadata
    analysis_duration_ms: float
    data_points_analyzed: int
    real_data_percentage: float
    
    def __post_init__(self):
        """Validate dimensional reading"""
        # CRITICAL: Ensure all data sources are real
        non_real_sources = [
            vr for vr in self.validation_results 
            if not vr.is_real or vr.source_type in [DataSourceType.SIMULATION, DataSourceType.MOCK]
        ]
        
        if non_real_sources:
            source_names = [vr.metadata.get('source_name', 'unknown') for vr in non_real_sources]
            raise ValueError(f"Non-real data sources in dimensional reading: {source_names}")
        
        # Validate signal strength
        if not -1.0 <= self.signal_strength <= 1.0:
            raise ValueError(f"Signal strength must be between -1.0 and 1.0, got {self.signal_strength}")
        
        # Ensure minimum real data percentage
        if self.real_data_percentage < 100.0:
            raise ValueError(f"Production requires 100% real data, got {self.real_data_percentage}%")

class ProductionDataValidator:
    """
    Production data validator that enforces real data usage
    """
    
    def __init__(self):
        self.simulation_patterns = [
            r'simulate.*\(',
            r'mock.*\(',
            r'generate.*random',
            r'_generate_simulated',
            r'fake.*data',
            r'dummy.*values',
            r'test.*data',
            r'synthetic.*',
            r'artificial.*'
        ]
    
    def validate_data_source(self, source_name: str, source_obj: Any) -> DataValidationResult:
        """
        Validate that data source is real and not simulated
        
        Args:
            source_name: Name of the data source
            source_obj: Data source object to validate
            
        Returns:
            DataValidationResult with validation status
        """
        validation_result = DataValidationResult(
            is_real=False,
            source_type=DataSourceType.SIMULATION,
            validation_timestamp=datetime.now(),
            metadata={'source_name': source_name}
        )
        
        # Check for simulation indicators in source name
        source_lower = source_name.lower()
        for pattern in ['simulate', 'mock', 'fake', 'test', 'dummy', 'synthetic']:
            if pattern in source_lower:
                validation_result.issues.append(f"Simulation indicator in source name: {pattern}")
                return validation_result
        
        # Check for simulation methods in source object
        if hasattr(source_obj, '__dict__'):
            for attr_name in dir(source_obj):
                if any(pattern.replace(r'\(', '').replace(r'.*', '') in attr_name.lower() 
                       for pattern in self.simulation_patterns):
                    validation_result.issues.append(f"Simulation method detected: {attr_name}")
                    return validation_result
        
        # Check for real API connectivity
        if hasattr(source_obj, 'test_connection'):
            try:
                # This should be an async call in real implementation
                connected = source_obj.test_connection()
                if not connected:
                    validation_result.issues.append("Failed to connect to real API")
                    return validation_result
            except Exception as e:
                validation_result.issues.append(f"Connection test failed: {e}")
                return validation_result
        else:
            validation_result.issues.append("No connection test method available")
            return validation_result
        
        # Check for real API credentials
        if hasattr(source_obj, 'api_key'):
            api_key = getattr(source_obj, 'api_key', '')
            if not api_key or api_key in ['demo', 'test', 'fake', 'mock']:
                validation_result.issues.append("Invalid or demo API key")
                return validation_result
        
        # If all checks pass, mark as real
        validation_result.is_real = True
        validation_result.source_type = DataSourceType.REAL_API
        validation_result.metadata['validated_at'] = datetime.now().isoformat()
        
        return validation_result
    
    def validate_market_data(self, market_data: ProductionMarketData) -> bool:
        """
        Validate market data for production use
        
        Args:
            market_data: Market data to validate
            
        Returns:
            True if data is valid for production use
        """
        # Check data source type
        if market_data.data_source_type in [DataSourceType.SIMULATION, DataSourceType.MOCK]:
            logger.error(f"Simulation/mock data rejected: {market_data.source}")
            return False
        
        # Check validation result
        if not market_data.validation_result.is_real:
            logger.error(f"Non-real data source rejected: {market_data.source}")
            return False
        
        # Check for data quality issues
        if market_data.validation_result.issues:
            logger.warning(f"Data quality issues: {market_data.validation_result.issues}")
            # Allow with warnings for now, but log for monitoring
        
        return True

class ProductionDimensionalEngine(ABC):
    """
    Abstract base class for production dimensional engines
    Enforces real data usage and prevents simulation
    """
    
    def __init__(self, dimension_name: str):
        self.dimension_name = dimension_name
        self.validator = ProductionDataValidator()
        self.data_sources = {}
        self.last_analysis_time = None
        self.analysis_count = 0
        
    def register_data_source(self, source_name: str, source_obj: Any) -> bool:
        """
        Register and validate a data source
        
        Args:
            source_name: Name of the data source
            source_obj: Data source object
            
        Returns:
            True if source is valid and registered
        """
        validation_result = self.validator.validate_data_source(source_name, source_obj)
        
        if not validation_result.is_real:
            logger.error(f"Rejecting non-real data source {source_name}: {validation_result.issues}")
            return False
        
        self.data_sources[source_name] = {
            'source': source_obj,
            'validation': validation_result,
            'registered_at': datetime.now()
        }
        
        logger.info(f"Registered real data source: {source_name}")
        return True
    
    @abstractmethod
    async def analyze(self, market_data: ProductionMarketData) -> ProductionDimensionalReading:
        """
        Perform dimensional analysis with real data
        
        Args:
            market_data: Production market data (validated as real)
            
        Returns:
            Dimensional reading based on real data analysis
        """
        pass
    
    def _validate_analysis_inputs(self, market_data: ProductionMarketData):
        """Validate inputs before analysis"""
        # Ensure market data is real
        if not self.validator.validate_market_data(market_data):
            raise ValueError("Invalid market data for production analysis")
        
        # Ensure we have real data sources
        if not self.data_sources:
            raise ValueError("No real data sources registered for analysis")
        
        # Check data source health
        for source_name, source_info in self.data_sources.items():
            age = datetime.now() - source_info['registered_at']
            if age > timedelta(hours=24):
                logger.warning(f"Data source {source_name} registration is {age} old")
    
    def _create_dimensional_reading(
        self, 
        signal_strength: float, 
        confidence: ConfidenceLevel,
        evidence: Dict[str, Any],
        analysis_duration_ms: float,
        data_points_count: int
    ) -> ProductionDimensionalReading:
        """Create validated dimensional reading"""
        
        # Collect validation results from all data sources
        validation_results = [
            source_info['validation'] 
            for source_info in self.data_sources.values()
        ]
        
        # Calculate real data percentage (should be 100% in production)
        real_data_percentage = 100.0  # All sources must be real
        
        return ProductionDimensionalReading(
            dimension=self.dimension_name,
            timestamp=datetime.now(),
            confidence=confidence,
            signal_strength=signal_strength,
            supporting_evidence=evidence,
            data_sources=list(self.data_sources.keys()),
            validation_results=validation_results,
            analysis_duration_ms=analysis_duration_ms,
            data_points_analyzed=data_points_count,
            real_data_percentage=real_data_percentage
        )

class ProductionSystemMonitor:
    """
    Monitor production system for simulation/mock data infiltration
    """
    
    def __init__(self):
        self.validator = ProductionDataValidator()
        self.monitoring_active = True
        self.violations = []
    
    async def monitor_data_flow(self, data_stream):
        """Monitor data stream for non-real data"""
        while self.monitoring_active:
            try:
                data_point = await data_stream.get_next()
                
                if isinstance(data_point, ProductionMarketData):
                    if not self.validator.validate_market_data(data_point):
                        violation = {
                            'timestamp': datetime.now(),
                            'type': 'invalid_market_data',
                            'source': data_point.source,
                            'details': data_point.validation_result.issues
                        }
                        self.violations.append(violation)
                        logger.error(f"Data validation violation: {violation}")
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(1)
    
    def get_violation_report(self) -> Dict[str, Any]:
        """Get report of all data validation violations"""
        return {
            'total_violations': len(self.violations),
            'violations': self.violations,
            'monitoring_since': datetime.now() - timedelta(hours=24),
            'system_status': 'COMPROMISED' if self.violations else 'CLEAN'
        }

# Production configuration validation
class ProductionConfig:
    """Production configuration with mandatory real API keys"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._validate_config()
    
    def _validate_config(self):
        """Validate production configuration"""
        required_real_keys = [
            'fred_api_key',
            'exchange_api_key',
            'price_data_api_key',
            'news_api_key'
        ]
        
        for key in required_real_keys:
            if key not in self.config:
                raise ValueError(f"Missing required API key: {key}")
            
            value = self.config[key]
            if not value or value in ['demo', 'test', 'fake', 'mock', 'placeholder']:
                raise ValueError(f"Invalid API key for {key}: must be real production key")
        
        # Validate environment
        environment = self.config.get('environment', 'development')
        if environment == 'production':
            # Additional production-specific validations
            if self.config.get('allow_simulation', False):
                raise ValueError("Simulation not allowed in production environment")
            
            if self.config.get('debug_mode', False):
                raise ValueError("Debug mode not allowed in production environment")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)

# Example usage
async def test_production_data_validation():
    """Test production data validation"""
    
    # This would fail - simulation data not allowed
    try:
        fake_validation = DataValidationResult(
            is_real=False,
            source_type=DataSourceType.SIMULATION,
            validation_timestamp=datetime.now()
        )
        
        fake_market_data = ProductionMarketData(
            symbol="EURUSD",
            timestamp=datetime.now(),
            bid=1.1000,
            ask=1.1002,
            volume=1000.0,
            source="simulated_feed",
            data_source_type=DataSourceType.SIMULATION,
            validation_result=fake_validation
        )
        
        print("ERROR: Simulation data was accepted!")
        
    except ValueError as e:
        print(f"SUCCESS: Simulation data correctly rejected: {e}")
    
    # This would pass - real data
    try:
        real_validation = DataValidationResult(
            is_real=True,
            source_type=DataSourceType.REAL_API,
            validation_timestamp=datetime.now()
        )
        
        real_market_data = ProductionMarketData(
            symbol="EURUSD",
            timestamp=datetime.now(),
            bid=1.1000,
            ask=1.1002,
            volume=1000.0,
            source="real_broker_feed",
            data_source_type=DataSourceType.REAL_API,
            validation_result=real_validation
        )
        
        print("SUCCESS: Real data correctly accepted")
        
    except ValueError as e:
        print(f"ERROR: Real data was rejected: {e}")

if __name__ == "__main__":
    asyncio.run(test_production_data_validation())

