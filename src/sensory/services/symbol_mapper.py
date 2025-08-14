"""
Dynamic Symbol Mapping Service

Provides broker-provided symbol mapping with Redis caching
to eliminate hard-coded symbol IDs.
"""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.core.event_bus import EventBus
from src.operational.state_store import StateStore

logger = logging.getLogger(__name__)


@dataclass
class SymbolInfo:
    """Information about a trading symbol."""
    symbol_id: int
    symbol_name: str
    description: str
    base_currency: str
    quote_currency: str
    precision: int
    pip_position: int
    contract_size: float
    min_volume: float
    max_volume: float
    step_volume: float
    swap_long: float
    swap_short: float
    margin_initial: float
    margin_maintenance: float


class SymbolMapper:
    """
    Manages dynamic symbol mapping from broker APIs.
    
    Features:
    - Fetches symbol information from broker
    - Caches mappings in Redis
    - Provides reverse lookup (name -> ID)
    - Automatic refresh on cache expiry
    """
    
    def __init__(self, event_bus: EventBus, state_store: StateStore):
        self.event_bus = event_bus
        self.state_store = state_store
        self._cache_key = "emp:symbol_map"
        self._cache_ttl = 3600  # 1 hour
        self._symbols: Dict[str, SymbolInfo] = {}
        
    async def initialize(self) -> None:
        """Initialize symbol mapping on startup."""
        await self.refresh_symbol_cache()
        
    async def refresh_symbol_cache(self) -> None:
        """Refresh symbol mappings from broker."""
        try:
            # Fetch symbols from broker (mock implementation)
            symbols = await self._fetch_symbols_from_broker()
            
            # Cache in Redis
            symbol_map = {
                symbol.symbol_name: {
                    'symbol_id': symbol.symbol_id,
                    'symbol_name': symbol.symbol_name,
                    'description': symbol.description,
                    'base_currency': symbol.base_currency,
                    'quote_currency': symbol.quote_currency,
                    'precision': symbol.precision,
                    'pip_position': symbol.pip_position,
                    'contract_size': symbol.contract_size,
                    'min_volume': symbol.min_volume,
                    'max_volume': symbol.max_volume,
                    'step_volume': symbol.step_volume,
                    'swap_long': symbol.swap_long,
                    'swap_short': symbol.swap_short,
                    'margin_initial': symbol.margin_initial,
                    'margin_maintenance': symbol.margin_maintenance
                }
                for symbol in symbols
            }
            
            await self.state_store.set(
                self._cache_key,
                json.dumps(symbol_map),
                expire=self._cache_ttl
            )
            
            # Update local cache
            self._symbols = {name: SymbolInfo(**info) for name, info in symbol_map.items()}
            
            logger.info(f"Refreshed symbol cache with {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to refresh symbol cache: {e}")
            # Fallback to cached data if available
            await self._load_from_cache()
    
    async def _fetch_symbols_from_broker(self) -> List[SymbolInfo]:
        """Fetch symbols from broker API (mock implementation)."""
        # In production, this would use actual broker API
        # For now, return mock data for common symbols
        return [
            SymbolInfo(
                symbol_id=1,
                symbol_name="EURUSD",
                description="Euro vs US Dollar",
                base_currency="EUR",
                quote_currency="USD",
                precision=5,
                pip_position=4,
                contract_size=100000,
                min_volume=0.01,
                max_volume=100,
                step_volume=0.01,
                swap_long=-0.5,
                swap_short=-0.3,
                margin_initial=0.01,
                margin_maintenance=0.005
            ),
            SymbolInfo(
                symbol_id=2,
                symbol_name="GBPUSD",
                description="British Pound vs US Dollar",
                base_currency="GBP",
                quote_currency="USD",
                precision=5,
                pip_position=4,
                contract_size=100000,
                min_volume=0.01,
                max_volume=100,
                step_volume=0.01,
                swap_long=-0.4,
                swap_short=-0.2,
                margin_initial=0.01,
                margin_maintenance=0.005
            ),
            SymbolInfo(
                symbol_id=3,
                symbol_name="USDJPY",
                description="US Dollar vs Japanese Yen",
                base_currency="USD",
                quote_currency="JPY",
                precision=3,
                pip_position=2,
                contract_size=100000,
                min_volume=0.01,
                max_volume=100,
                step_volume=0.01,
                swap_long=0.1,
                swap_short=-0.6,
                margin_initial=0.01,
                margin_maintenance=0.005
            )
        ]
    
    async def _load_from_cache(self) -> None:
        """Load symbol mappings from Redis cache."""
        try:
            cached_data = await self.state_store.get(self._cache_key)
            if cached_data:
                symbol_map = json.loads(cached_data)
                self._symbols = {name: SymbolInfo(**info) for name, info in symbol_map.items()}
                logger.info(f"Loaded {len(self._symbols)} symbols from cache")
            else:
                logger.warning("No cached symbol data available")
        except Exception as e:
            logger.error(f"Failed to load from cache: {e}")
    
    def get_symbol_id(self, symbol_name: str) -> Optional[int]:
        """Get symbol ID by name."""
        symbol = self._symbols.get(symbol_name)
        return symbol.symbol_id if symbol else None
    
    def get_symbol_info(self, symbol_name: str) -> Optional[SymbolInfo]:
        """Get complete symbol information."""
        return self._symbols.get(symbol_name)
    
    def get_all_symbols(self) -> List[str]:
        """Get list of all available symbol names."""
        return list(self._symbols.keys())
    
    async def get_symbol_info_async(self, symbol_name: str) -> Optional[SymbolInfo]:
        """Get symbol info with cache refresh if needed."""
        if symbol_name not in self._symbols:
            await self.refresh_symbol_cache()
        return self._symbols.get(symbol_name)


# Global instance
_symbol_mapper: Optional[SymbolMapper] = None


async def get_symbol_mapper(event_bus: EventBus, state_store: StateStore) -> SymbolMapper:
    """Get or create global symbol mapper instance."""
    global _symbol_mapper
    if _symbol_mapper is None:
        _symbol_mapper = SymbolMapper(event_bus, state_store)
        await _symbol_mapper.initialize()
    return _symbol_mapper
