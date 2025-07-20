"""
Core models for the EMP system.

This module contains shared data models used across the system.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import numpy as np


@dataclass
class InstrumentMeta:
    """Represents trading instrument metadata."""
    symbol: str
    pip_size: float
    lot_size: int
    commission: float = 0.0
    spread: float = 0.0
    leverage: int = 50


@dataclass
class TokenData:
    """Represents OAuth token data."""
    access_token: str
    expires_in: int
    refresh_token: str = ""
    scope: str = ""
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class MarketContext:
    """Represents a market context for pattern memory."""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    volatility: float
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyModel:
    """Represents a trading strategy in the database."""
    genome_id: str
    dna: str
    fitness_score: float = 0.0
    generation: int = 0
    is_champion: bool = False
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


@dataclass
class ContextPacket:
    """Enhanced context packet with memory support."""
    symbol: str
    timestamp: datetime
    latent_vector: np.ndarray
    market_data: Dict[str, Any]
    memory_match: Optional[Tuple[List[float], List[int]]] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# Update ContextPacket to include memory_match
ContextPacket.__annotations__['memory_match'] = Optional[Tuple[List[float], List[int]]]
