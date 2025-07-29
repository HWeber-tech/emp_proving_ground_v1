"""
World-Class Validation Framework
Comprehensive validation utilities for production trading system.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass

from .exceptions import ValidationException, DataValidationException


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    context: Dict[str, Any]
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.context is None:
            self.context = {}


class BaseValidator:
    """Base class for all validators."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate a value and return detailed results."""
        raise NotImplementedError("Subclasses must implement validate method")
    
    def _create_result(self, is_valid: bool, errors: List[str] = None, 
                      warnings: List[str] = None, context: Dict[str, Any] = None) -> ValidationResult:
        """Create a validation result."""
        return ValidationResult(
            is_valid=is_valid,
            errors=errors or [],
            warnings=warnings or [],
            context=context or {}
        )


class TradingValidator(BaseValidator):
    """Validator for trading-related data."""
    
    def __init__(self):
        super().__init__("TradingValidator")
    
    def validate_symbol(self, symbol: str) -> ValidationResult:
        """Validate trading symbol format."""
        errors = []
        warnings = []
        
        if not symbol:
            errors.append("Symbol cannot be empty")
        elif not isinstance(symbol, str):
            errors.append("Symbol must be a string")
        else:
            # Check symbol format (basic validation)
            if len(symbol) < 2:
                errors.append("Symbol must be at least 2 characters")
            if len(symbol) > 12:
                warnings.append("Symbol is unusually long")
            if not re.match(r'^[A-Z0-9._-]+$', symbol.upper()):
                errors.append("Symbol contains invalid characters")
        
        return self._create_result(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            context={'symbol': symbol}
        )
    
    def validate_price(self, price: Union[float, Decimal], symbol: str = None) -> ValidationResult:
        """Validate price value."""
        errors = []
        warnings = []
        
        if price is None:
            errors.append("Price cannot be None")
        elif not isinstance(price, (int, float, Decimal)):
            errors.append("Price must be numeric")
        else:
            price_val = float(price)
            if price_val <= 0:
                errors.append("Price must be positive")
            if price_val > 1000000:
                warnings.append("Price is unusually high")
            if price_val < 0.00001:
                warnings.append("Price is unusually low")
        
        return self._create_result(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            context={'price': price, 'symbol': symbol}
        )
    
    def validate_quantity(self, quantity: Union[int, float], symbol: str = None) -> ValidationResult:
        """Validate order quantity."""
        errors = []
        warnings = []
        
        if quantity is None:
            errors.append("Quantity cannot be None")
        elif not isinstance(quantity, (int, float)):
            errors.append("Quantity must be numeric")
        else:
            if quantity <= 0:
                errors.append("Quantity must be positive")
            if quantity > 10000000:
                warnings.append("Quantity is unusually large")
        
        return self._create_result(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            context={'quantity': quantity, 'symbol': symbol}
        )
    
    def validate_order_data(self, order_data: Dict[str, Any]) -> ValidationResult:
        """Validate complete order data."""
        errors = []
        warnings = []
        
        required_fields = ['symbol', 'side', 'quantity']
        for field in required_fields:
            if field not in order_data:
                errors.append(f"Missing required field: {field}")
        
        if 'symbol' in order_data:
            symbol_result = self.validate_symbol(order_data['symbol'])
            errors.extend(symbol_result.errors)
            warnings.extend(symbol_result.warnings)
        
        if 'quantity' in order_data:
            quantity_result = self.validate_quantity(order_data['quantity'])
            errors.extend(quantity_result.errors)
            warnings.extend(quantity_result.warnings)
        
        if 'price' in order_data and order_data['price'] is not None:
            price_result = self.validate_price(order_data['price'])
            errors.extend(price_result.errors)
            warnings.extend(price_result.warnings)
        
        if 'side' in order_data:
            if order_data['side'] not in ['BUY', 'SELL']:
                errors.append("Side must be 'BUY' or 'SELL'")
        
        return self._create_result(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            context=order_data
        )


class MarketDataValidator(BaseValidator):
    """Validator for market data."""
    
    def __init__(self):
        super().__init__("MarketDataValidator")
    
    def validate_ohlc_data(self, ohlc_data: Dict[str, float]) -> ValidationResult:
        """Validate OHLC market data."""
        errors = []
        warnings = []
        
        required_fields = ['open', 'high', 'low', 'close']
        for field in required_fields:
            if field not in ohlc_data:
                errors.append(f"Missing OHLC field: {field}")
        
        if len(errors) == 0:
            o, h, l, c = ohlc_data['open'], ohlc_data['high'], ohlc_data['low'], ohlc_data['close']
            
            # Validate OHLC relationships
            if h < max(o, c):
                errors.append("High must be >= max(open, close)")
            if l > min(o, c):
                errors.append("Low must be <= min(open, close)")
            if any(val <= 0 for val in [o, h, l, c]):
                errors.append("All OHLC values must be positive")
            
            # Check for unusual price movements
            if abs(c - o) / o > 0.2:  # 20% change
                warnings.append("Unusual price movement detected")
        
        return self._create_result(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            context=ohlc_data
        )
    
    def validate_timestamp(self, timestamp: Union[datetime, str, int]) -> ValidationResult:
        """Validate timestamp format and reasonableness."""
        errors = []
        warnings = []
        
        try:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            elif isinstance(timestamp, int):
                dt = datetime.fromtimestamp(timestamp)
            elif isinstance(timestamp, datetime):
                dt = timestamp
            else:
                errors.append("Invalid timestamp format")
                dt = None
            
            if dt:
                now = datetime.utcnow()
                if dt > now + timedelta(hours=1):
                    errors.append("Timestamp is in the future")
                if dt < now - timedelta(days=365):
                    warnings.append("Timestamp is more than 1 year old")
                
        except (ValueError, TypeError) as e:
            errors.append(f"Invalid timestamp: {e}")
        
        return self._create_result(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            context={'timestamp': timestamp}
        )


class GenomeValidator(BaseValidator):
    """Validator for genetic algorithm genomes."""
    
    def __init__(self):
        super().__init__("GenomeValidator")
    
    def validate_genome_parameters(self, parameters: Dict[str, Any]) -> ValidationResult:
        """Validate genome parameters."""
        errors = []
        warnings = []
        
        if not parameters:
            errors.append("Genome parameters cannot be empty")
            return self._create_result(False, errors, warnings, {'parameters': parameters})
        
        # Validate parameter ranges
        parameter_ranges = {
            'risk_tolerance': (0.0, 1.0),
            'position_size_factor': (0.01, 1.0),
            'stop_loss_factor': (0.01, 0.5),
            'take_profit_factor': (1.1, 10.0),
            'momentum_threshold': (0.0, 1.0),
            'volatility_threshold': (0.0, 1.0)
        }
        
        for param, value in parameters.items():
            if param in parameter_ranges:
                min_val, max_val = parameter_ranges[param]
                if not isinstance(value, (int, float)):
                    errors.append(f"Parameter {param} must be numeric")
                elif not (min_val <= value <= max_val):
                    errors.append(f"Parameter {param} must be between {min_val} and {max_val}")
        
        # Check for required parameters
        required_params = ['risk_tolerance', 'position_size_factor']
        for param in required_params:
            if param not in parameters:
                errors.append(f"Missing required parameter: {param}")
        
        return self._create_result(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            context={'parameters': parameters}
        )
    
    def validate_fitness_score(self, fitness: float) -> ValidationResult:
        """Validate fitness score."""
        errors = []
        warnings = []
        
        if not isinstance(fitness, (int, float)):
            errors.append("Fitness must be numeric")
        elif fitness < 0:
            errors.append("Fitness cannot be negative")
        elif fitness > 100:
            warnings.append("Unusually high fitness score")
        
        return self._create_result(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            context={'fitness': fitness}
        )


class SystemValidator(BaseValidator):
    """Validator for system-level components."""
    
    def __init__(self):
        super().__init__("SystemValidator")
    
    def validate_configuration(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate system configuration."""
        errors = []
        warnings = []
        
        # Validate required configuration sections
        required_sections = ['trading', 'data', 'evolution']
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing configuration section: {section}")
        
        # Validate trading configuration
        if 'trading' in config:
            trading_config = config['trading']
            if 'broker' not in trading_config:
                errors.append("Missing broker configuration")
            if 'account_id' not in trading_config:
                errors.append("Missing account_id in trading configuration")
        
        return self._create_result(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            context=config
        )


class ValidationManager:
    """Central manager for all validation operations."""
    
    def __init__(self):
        self.validators = {
            'trading': TradingValidator(),
            'market_data': MarketDataValidator(),
            'genome': GenomeValidator(),
            'system': SystemValidator()
        }
        self.logger = logging.getLogger(__name__)
    
    def validate(self, validator_type: str, data: Any, **kwargs) -> ValidationResult:
        """Perform validation using specified validator."""
        if validator_type not in self.validators:
            raise ValidationException(f"Unknown validator type: {validator_type}")
        
        validator = self.validators[validator_type]
        
        # Route to appropriate validation method based on data type
        if validator_type == 'trading':
            if isinstance(data, dict):
                return validator.validate_order_data(data)
            elif isinstance(data, str):
                return validator.validate_symbol(data)
        elif validator_type == 'market_data':
            if isinstance(data, dict) and 'open' in data:
                return validator.validate_ohlc_data(data)
        elif validator_type == 'genome':
            if isinstance(data, dict):
                return validator.validate_genome_parameters(data)
            elif isinstance(data, (int, float)):
                return validator.validate_fitness_score(data)
        elif validator_type == 'system':
            return validator.validate_configuration(data)
        
        # Fallback to generic validation
        return validator.validate(data, kwargs)
    
    def validate_and_raise(self, validator_type: str, data: Any, **kwargs):
        """Validate data and raise exception if invalid."""
        result = self.validate(validator_type, data, **kwargs)
        if not result.is_valid:
            error_msg = f"Validation failed: {'; '.join(result.errors)}"
            raise ValidationException(error_msg, context=result.context)
        return result
    
    def get_validator(self, validator_type: str) -> BaseValidator:
        """Get specific validator instance."""
        if validator_type not in self.validators:
            raise ValidationException(f"Unknown validator type: {validator_type}")
        return self.validators[validator_type]


# Global validation manager instance
validation_manager = ValidationManager()


# Validation decorators
def validate_input(validator_type: str, param_name: str = None):
    """Decorator to validate function input parameters."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Determine which parameter to validate
            if param_name and param_name in kwargs:
                data = kwargs[param_name]
            elif len(args) > 0:
                data = args[0]
            else:
                raise ValidationException("No data to validate")
            
            # Perform validation
            result = validation_manager.validate(validator_type, data)
            if not result.is_valid:
                error_msg = f"Input validation failed in {func.__name__}: {'; '.join(result.errors)}"
                raise ValidationException(error_msg, context=result.context)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_trading_data(func):
    """Decorator for trading data validation."""
    return validate_input('trading')(func)


def validate_market_data(func):
    """Decorator for market data validation."""
    return validate_input('market_data')(func)


def validate_genome_data(func):
    """Decorator for genome data validation."""
    return validate_input('genome')(func)

