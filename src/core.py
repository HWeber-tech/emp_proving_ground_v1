"""
Core foundational components for the EMP Proving Ground system.

This module contains the essential building blocks:
- RiskConfig: Configuration management for risk parameters
- Instrument: Financial instrument metadata
- InstrumentProvider: Instrument data management
- CurrencyConverter: Currency conversion and pip value calculations
"""

import json
import logging
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Canonical RiskConfig is defined in src.config.risk.risk_config


# Re-export canonical Instrument implementation to avoid duplicate class definitions
from src.core.instrument import Instrument as Instrument


class InstrumentProvider:
    """Manages instrument metadata and provides access to instrument data"""
    
    def __init__(self, instruments_file: str = "config/system/instruments.json"):
        self.instruments_file = Path(instruments_file)
        self.instruments: Dict[str, Instrument] = {}
        self._load_instruments()
    
    def _load_instruments(self):
        """Load instruments from JSON file or create defaults"""
        if self.instruments_file.exists():
            try:
                with open(self.instruments_file, 'r') as f:
                    data = json.load(f)
                
                for symbol, config in data.items():
                    self.instruments[symbol] = Instrument(
                        symbol=symbol,
                        pip_decimal_places=config['pip_decimal_places'],
                        contract_size=Decimal(str(config['contract_size'])),
                        long_swap_rate=Decimal(str(config['long_swap_rate'])),
                        short_swap_rate=Decimal(str(config['short_swap_rate'])),
                        margin_currency=config['margin_currency'],
                        swap_time=config.get('swap_time', '22:00')
                    )
                logger.info(f"Loaded {len(self.instruments)} instruments from {self.instruments_file}")
            except Exception as e:
                logger.warning(f"Failed to load instruments from {self.instruments_file}: {e}")
                self._create_default_instruments()
        else:
            self._create_default_instruments()
    
    def _create_default_instruments(self):
        """Create default instrument configurations"""
        default_instruments = {
            "EURUSD": {
                "pip_decimal_places": 4,
                "contract_size": "100000",
                "long_swap_rate": "-0.0001",
                "short_swap_rate": "0.0001",
                "margin_currency": "USD",
                "swap_time": "22:00"
            },
            "GBPUSD": {
                "pip_decimal_places": 4,
                "contract_size": "100000",
                "long_swap_rate": "-0.0002",
                "short_swap_rate": "0.0002",
                "margin_currency": "USD",
                "swap_time": "22:00"
            },
            "USDJPY": {
                "pip_decimal_places": 2,
                "contract_size": "100000",
                "long_swap_rate": "-0.0001",
                "short_swap_rate": "0.0001",
                "margin_currency": "USD",
                "swap_time": "22:00"
            }
        }
        
        for symbol, config in default_instruments.items():
            self.instruments[symbol] = Instrument(
                symbol=symbol,
                pip_decimal_places=config['pip_decimal_places'],
                contract_size=Decimal(config['contract_size']),
                long_swap_rate=Decimal(config['long_swap_rate']),
                short_swap_rate=Decimal(config['short_swap_rate']),
                margin_currency=config['margin_currency'],
                swap_time=config['swap_time']
            )
        
        # Save default instruments
        self._save_instruments()
        logger.info(f"Created {len(self.instruments)} default instruments")
    
    def _save_instruments(self):
        """Save instruments to JSON file"""
        try:
            self.instruments_file.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            for symbol, instrument in self.instruments.items():
                data[symbol] = {
                    'pip_decimal_places': instrument.pip_decimal_places,
                    'contract_size': str(instrument.contract_size),
                    'long_swap_rate': str(instrument.long_swap_rate),
                    'short_swap_rate': str(instrument.short_swap_rate),
                    'margin_currency': instrument.margin_currency,
                    'swap_time': instrument.swap_time
                }
            
            with open(self.instruments_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save instruments: {e}")
    
    def get_instrument(self, symbol: str) -> Optional[Instrument]:
        """Get instrument by symbol"""
        return self.instruments.get(symbol.upper())
    
    def list_instruments(self) -> List[str]:
        """List all available instrument symbols"""
        return list(self.instruments.keys())


class CurrencyConverter:
    """Handles currency conversion and pip value calculations"""
    
    def __init__(self, rates_file: str = "config/system/exchange_rates.json"):
        self.rates_file = Path(rates_file)
        self.rates: Dict[str, Dict[str, float]] = {}
        self._load_rates()
    
    def _load_rates(self):
        """Load exchange rates from JSON file or create defaults"""
        if self.rates_file.exists():
            try:
                with open(self.rates_file, 'r') as f:
                    self.rates = json.load(f)
                logger.info(f"Loaded exchange rates from {self.rates_file}")
            except Exception as e:
                logger.warning(f"Failed to load exchange rates: {e}")
                self._create_default_rates()
        else:
            self._create_default_rates()
    
    def _create_default_rates(self):
        """Create default exchange rates"""
        self.rates = {
            "USD": {
                "EUR": 0.85,
                "GBP": 0.73,
                "JPY": 110.0,
                "CHF": 0.92,
                "AUD": 1.35,
                "CAD": 1.25,
                "NZD": 1.40
            },
            "EUR": {
                "USD": 1.18,
                "GBP": 0.86,
                "JPY": 129.4,
                "CHF": 1.08,
                "AUD": 1.59,
                "CAD": 1.47,
                "NZD": 1.65
            }
        }
        
        # Add reverse rates
        for base, quotes in list(self.rates.items()):
            for quote, rate in quotes.items():
                if quote not in self.rates:
                    self.rates[quote] = {}
                self.rates[quote][base] = 1.0 / rate
        
        # Add base currency rates
        for currency in self.rates:
            self.rates[currency][currency] = 1.0
        
        self._save_rates()
        logger.info("Created default exchange rates")
    
    def _save_rates(self):
        """Save exchange rates to JSON file"""
        try:
            self.rates_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.rates_file, 'w') as f:
                json.dump(self.rates, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save exchange rates: {e}")
    
    def get_rate(self, from_currency: str, to_currency: str) -> Decimal:
        """Get exchange rate between two currencies"""
        from_curr = from_currency.upper()
        to_curr = to_currency.upper()
        
        if from_curr == to_curr:
            return Decimal('1.0')
        
        if from_curr in self.rates and to_curr in self.rates[from_curr]:
            return Decimal(str(self.rates[from_curr][to_curr]))
        
        # Try reverse rate
        if to_curr in self.rates and from_curr in self.rates[to_curr]:
            return Decimal(str(1.0 / self.rates[to_curr][from_curr]))
        
        raise ValueError(f"No exchange rate found for {from_curr}/{to_curr}")
    
    def calculate_pip_value(self, instrument: Instrument, account_currency: str) -> Decimal:
        """
        Calculate the pip value in account currency
        
        Args:
            instrument: The trading instrument
            account_currency: The account's base currency
            
        Returns:
            Pip value in account currency
        """
        # Get the quote currency from the symbol (e.g., USD from EURUSD)
        quote_currency = instrument.symbol[3:6] if len(instrument.symbol) >= 6 else "USD"
        
        # Calculate pip value in quote currency
        pip_value_quote = instrument.contract_size * Decimal('0.0001')
        if instrument.pip_decimal_places == 2:  # JPY pairs
            pip_value_quote = instrument.contract_size * Decimal('0.01')
        
        # Convert to account currency
        if quote_currency != account_currency:
            rate = self.get_rate(quote_currency, account_currency)
            return pip_value_quote * rate
        else:
            return pip_value_quote 
