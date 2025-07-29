"""
Data Validation Module - Phase 1 Implementation

This module provides comprehensive data validation for real data sources.
It ensures data quality, consistency, and reliability.

Author: EMP Development Team
Date: July 18, 2024
Phase: 1 - Real Data Foundation
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from src.sensory.core.base import MarketData

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation levels"""
    BASIC = "basic"
    STRICT = "strict"
    LENIENT = "lenient"


class DataIssue(Enum):
    """Types of data issues"""
    MISSING_DATA = "missing_data"
    OUTLIER = "outlier"
    INCONSISTENT = "inconsistent"
    STALE = "stale"
    INVALID_FORMAT = "invalid_format"
    DUPLICATE = "duplicate"
    ZERO_VOLUME = "zero_volume"
    NEGATIVE_PRICE = "negative_price"
    EXTREME_VOLATILITY = "extreme_volatility"


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    issues: List[DataIssue]
    confidence: float  # 0-1, confidence in data quality
    details: Dict[str, Any]
    timestamp: datetime


@dataclass
class DataQualityThresholds:
    """Thresholds for data quality validation"""
    max_age_minutes: int = 5
    min_volume: float = 0.0
    max_price_change_pct: float = 50.0
    max_volatility: float = 0.5
    min_confidence: float = 0.7
    outlier_threshold: float = 3.0  # Standard deviations


class MarketDataValidator:
    """Validator for market data"""
    
    def __init__(self, thresholds: Optional[DataQualityThresholds] = None):
        self.thresholds = thresholds or DataQualityThresholds()
        self.validation_history = []
        
    def validate_market_data(self, data: MarketData, level: ValidationLevel = ValidationLevel.STRICT) -> ValidationResult:
        """Validate market data"""
        issues = []
        details = {}
        
        # Basic validation
        basic_issues, basic_details = self._basic_validation(data)
        issues.extend(basic_issues)
        details.update(basic_details)
        
        # Level-specific validation
        if level == ValidationLevel.STRICT:
            strict_issues, strict_details = self._strict_validation(data)
            issues.extend(strict_issues)
            details.update(strict_details)
        elif level == ValidationLevel.LENIENT:
            lenient_issues, lenient_details = self._lenient_validation(data)
            issues.extend(lenient_issues)
            details.update(lenient_details)
        
        # Calculate confidence
        confidence = self._calculate_confidence(issues, details)
        
        # Determine validity
        is_valid = len(issues) == 0 and confidence >= self.thresholds.min_confidence
        
        result = ValidationResult(
            is_valid=is_valid,
            issues=issues,
            confidence=confidence,
            details=details,
            timestamp=datetime.now()
        )
        
        # Store in history
        self.validation_history.append(result)
        
        return result
    
    def _basic_validation(self, data: MarketData) -> Tuple[List[DataIssue], Dict[str, Any]]:
        """Basic validation checks"""
        issues = []
        details = {}
        
        # Check for missing data
        if data.bid is None or data.ask is None or data.volume is None:
            issues.append(DataIssue.MISSING_DATA)
            details['missing_fields'] = []
            if data.bid is None:
                details['missing_fields'].append('bid')
            if data.ask is None:
                details['missing_fields'].append('ask')
            if data.volume is None:
                details['missing_fields'].append('volume')
        
        # Check for negative prices
        if data.bid is not None and data.bid <= 0:
            issues.append(DataIssue.NEGATIVE_PRICE)
            details['negative_bid'] = data.bid
        
        if data.ask is not None and data.ask <= 0:
            issues.append(DataIssue.NEGATIVE_PRICE)
            details['negative_ask'] = data.ask
        
        # Check for zero volume
        if data.volume is not None and data.volume <= self.thresholds.min_volume:
            issues.append(DataIssue.ZERO_VOLUME)
            details['zero_volume'] = data.volume
        
        # Check for stale data
        if data.timestamp:
            age_minutes = (datetime.now() - data.timestamp).total_seconds() / 60
            if age_minutes > self.thresholds.max_age_minutes:
                issues.append(DataIssue.STALE)
                details['age_minutes'] = age_minutes
        
        # Check bid-ask spread
        if data.bid is not None and data.ask is not None:
            spread = data.ask - data.bid
            spread_pct = (spread / data.bid) * 100
            
            if spread_pct > 10:  # More than 10% spread is suspicious
                issues.append(DataIssue.INCONSISTENT)
                details['large_spread_pct'] = spread_pct
        
        return issues, details
    
    def _strict_validation(self, data: MarketData) -> Tuple[List[DataIssue], Dict[str, Any]]:
        """Strict validation checks"""
        issues = []
        details = {}
        
        # Check for extreme volatility
        if data.volatility is not None and data.volatility > self.thresholds.max_volatility:
            issues.append(DataIssue.EXTREME_VOLATILITY)
            details['extreme_volatility'] = data.volatility
        
        # Check for extreme price changes (if we have historical context)
        if hasattr(self, 'last_price') and self.last_price:
            price_change_pct = abs((data.bid - self.last_price) / self.last_price) * 100
            if price_change_pct > self.thresholds.max_price_change_pct:
                issues.append(DataIssue.OUTLIER)
                details['extreme_price_change_pct'] = price_change_pct
        
        # Update last price for next validation
        if data.bid is not None:
            self.last_price = data.bid
        
        return issues, details
    
    def _lenient_validation(self, data: MarketData) -> Tuple[List[DataIssue], Dict[str, Any]]:
        """Lenient validation checks"""
        issues = []
        details = {}
        
        # Only check for critical issues
        if data.bid is None or data.ask is None:
            issues.append(DataIssue.MISSING_DATA)
            details['critical_missing'] = True
        
        if data.bid is not None and data.bid <= 0:
            issues.append(DataIssue.NEGATIVE_PRICE)
            details['critical_negative_price'] = True
        
        return issues, details
    
    def _calculate_confidence(self, issues: List[DataIssue], details: Dict[str, Any]) -> float:
        """Calculate confidence score based on issues"""
        if not issues:
            return 1.0
        
        # Weight different types of issues
        issue_weights = {
            DataIssue.MISSING_DATA: 0.8,
            DataIssue.NEGATIVE_PRICE: 0.9,
            DataIssue.ZERO_VOLUME: 0.3,
            DataIssue.STALE: 0.5,
            DataIssue.INCONSISTENT: 0.6,
            DataIssue.OUTLIER: 0.4,
            DataIssue.EXTREME_VOLATILITY: 0.7,
            DataIssue.INVALID_FORMAT: 0.8,
            DataIssue.DUPLICATE: 0.2
        }
        
        # Calculate weighted penalty
        total_penalty = 0.0
        for issue in issues:
            weight = issue_weights.get(issue, 0.5)
            total_penalty += weight
        
        # Normalize penalty
        max_penalty = sum(issue_weights.values())
        normalized_penalty = min(total_penalty / max_penalty, 1.0)
        
        # Convert to confidence (1.0 = perfect, 0.0 = terrible)
        confidence = 1.0 - normalized_penalty
        
        return max(0.0, confidence)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of recent validations"""
        if not self.validation_history:
            return {}
        
        recent_validations = self.validation_history[-100:]  # Last 100 validations
        
        total_validations = len(recent_validations)
        valid_count = sum(1 for v in recent_validations if v.is_valid)
        avg_confidence = np.mean([v.confidence for v in recent_validations])
        
        # Count issues by type
        issue_counts = {}
        for validation in recent_validations:
            for issue in validation.issues:
                issue_counts[issue.value] = issue_counts.get(issue.value, 0) + 1
        
        return {
            'total_validations': total_validations,
            'valid_count': valid_count,
            'valid_percentage': (valid_count / total_validations) * 100 if total_validations > 0 else 0,
            'average_confidence': avg_confidence,
            'issue_breakdown': issue_counts
        }


class DataConsistencyChecker:
    """Check for data consistency across sources"""
    
    def __init__(self):
        self.source_data = {}
        self.consistency_threshold = 0.05  # 5% difference threshold
        
    def add_source_data(self, source: str, data: MarketData):
        """Add data from a source for consistency checking"""
        self.source_data[source] = data
    
    def check_consistency(self) -> Dict[str, Any]:
        """Check consistency across all sources"""
        if len(self.source_data) < 2:
            return {'consistent': True, 'sources': len(self.source_data)}
        
        sources = list(self.source_data.keys())
        prices = []
        
        for source in sources:
            data = self.source_data[source]
            if data.bid is not None:
                prices.append((source, data.bid))
        
        if len(prices) < 2:
            return {'consistent': True, 'sources': len(prices)}
        
        # Calculate price differences
        differences = []
        for i in range(len(prices)):
            for j in range(i + 1, len(prices)):
                source1, price1 = prices[i]
                source2, price2 = prices[j]
                
                diff_pct = abs(price1 - price2) / min(price1, price2) * 100
                differences.append({
                    'source1': source1,
                    'source2': source2,
                    'price1': price1,
                    'price2': price2,
                    'difference_pct': diff_pct
                })
        
        # Check if any differences exceed threshold
        inconsistent_pairs = [d for d in differences if d['difference_pct'] > self.consistency_threshold]
        
        return {
            'consistent': len(inconsistent_pairs) == 0,
            'sources': len(sources),
            'differences': differences,
            'inconsistent_pairs': inconsistent_pairs,
            'threshold_pct': self.consistency_threshold
        }
    
    def clear_data(self):
        """Clear stored data"""
        self.source_data.clear()


class DataQualityMonitor:
    """Monitor data quality over time"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.quality_metrics = []
        self.alert_threshold = 0.8
        
    def add_quality_metric(self, metric: Dict[str, Any]):
        """Add a quality metric"""
        self.quality_metrics.append({
            **metric,
            'timestamp': datetime.now()
        })
        
        # Keep only recent metrics
        if len(self.quality_metrics) > self.window_size:
            self.quality_metrics.pop(0)
    
    def get_quality_trend(self) -> Dict[str, Any]:
        """Get quality trend over time"""
        if not self.quality_metrics:
            return {}
        
        # Calculate trends
        confidences = [m.get('confidence', 0) for m in self.quality_metrics]
        valid_rates = [m.get('valid_rate', 0) for m in self.quality_metrics]
        
        if not confidences:
            return {}
        
        return {
            'average_confidence': np.mean(confidences),
            'confidence_trend': self._calculate_trend(confidences),
            'average_valid_rate': np.mean(valid_rates) if valid_rates else 0,
            'valid_rate_trend': self._calculate_trend(valid_rates) if valid_rates else 'stable',
            'total_metrics': len(self.quality_metrics),
            'alert_level': self._calculate_alert_level(confidences)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_alert_level(self, confidences: List[float]) -> str:
        """Calculate alert level based on recent confidence"""
        if not confidences:
            return 'unknown'
        
        recent_confidence = np.mean(confidences[-10:])  # Last 10 metrics
        
        if recent_confidence < self.alert_threshold:
            return 'critical'
        elif recent_confidence < 0.9:
            return 'warning'
        else:
            return 'normal'
    
    def should_alert(self) -> bool:
        """Check if quality alert should be triggered"""
        trend = self.get_quality_trend()
        return trend.get('alert_level', 'normal') in ['critical', 'warning']


# Example usage
def test_data_validation():
    """Test data validation functionality"""
    
    # Create validator
    thresholds = DataQualityThresholds(
        max_age_minutes=5,
        min_volume=100,
        max_price_change_pct=20,
        max_volatility=0.3,
        min_confidence=0.7
    )
    
    validator = MarketDataValidator(thresholds)
    
    # Test valid data
    valid_data = MarketData(
        timestamp=datetime.now(),
        bid=1.0950,
        ask=1.0952,
        volume=1000,
        volatility=0.01
    )
    
    result = validator.validate_market_data(valid_data)
    print(f"Valid data result: {result.is_valid}, confidence: {result.confidence:.3f}")
    
    # Test invalid data
    invalid_data = MarketData(
        timestamp=datetime.now() - timedelta(minutes=10),  # Stale
        bid=-1.0,  # Negative price
        ask=1.0952,
        volume=0,  # Zero volume
        volatility=0.8  # Extreme volatility
    )
    
    result = validator.validate_market_data(invalid_data)
    print(f"Invalid data result: {result.is_valid}, confidence: {result.confidence:.3f}")
    print(f"Issues: {[issue.value for issue in result.issues]}")
    
    # Test consistency checker
    checker = DataConsistencyChecker()
    
    data1 = MarketData(timestamp=datetime.now(), bid=1.0950, ask=1.0952, volume=1000, volatility=0.01)
    data2 = MarketData(timestamp=datetime.now(), bid=1.0951, ask=1.0953, volume=1000, volatility=0.01)
    
    checker.add_source_data("yahoo", data1)
    checker.add_source_data("alpha_vantage", data2)
    
    consistency = checker.check_consistency()
    print(f"Consistency check: {consistency}")


if __name__ == "__main__":
    test_data_validation() 
