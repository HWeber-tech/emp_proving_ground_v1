"""
Sensory Cortex v2.2 - Core Base Classes
try:
    from src.core.market_data import MarketData  # legacy
except Exception:  # pragma: no cover
    MarketData = object  # type: ignore

Canonical dataclasses that serve as the common language for the entire sensory system.
These provide type safety, clear data contracts, and eliminate the need for ad-hoc data structures.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, root_validator, validator
from src.core.base import MarketRegime as MarketRegime

from src.trading.order_management.order_book.snapshot import OrderBookLevel as OrderBookLevel
from src.trading.order_management.order_book.snapshot import OrderBookSnapshot as OrderBookSnapshot

logger = logging.getLogger(__name__)




class ConfidenceLevel(Enum):
    """Standardized confidence levels across all dimensions"""

    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9


class EventTier(Enum):
    """Economic event importance tiers"""

    TIER_1 = "tier_1"  # High impact (NFP, FOMC, CPI)
    TIER_2 = "tier_2"  # Medium impact (PMI, Retail Sales)
    TIER_3 = "tier_3"  # Low impact (Housing data, minor indicators)


class SessionType(Enum):
    """Trading session types"""

    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP_LONDON_NY = "overlap_london_ny"
    DEAD_ZONE = "dead_zone"


class SensoryReading(BaseModel):
    """Standardized sensory reading from any organ."""

    organ_name: str
    timestamp: datetime
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SensoryOrgan(ABC):
    """Base class for all sensory organs."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"sensory.{name}")

    @abstractmethod
    async def process(self, market_data: "MarketData") -> SensoryReading:
        """Process market data and return sensory reading."""
        pass

    def _create_error_reading(self, timestamp: datetime) -> SensoryReading:
        """Create error reading when processing fails."""
        return SensoryReading(
            organ_name=self.name,
            timestamp=timestamp,
            data={},
            metadata={"error": "Processing failed", "organ_version": "1.1.0"},
        )


class InstrumentMeta(BaseModel):
    """
    Canonical instrument metadata that all dimensional engines can rely on.
    Provides essential trading parameters and session information.
    """

    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    pip_size: float = Field(..., description="Pip size for the instrument")
    lot_size: int = Field(100000, description="Standard lot size")
    timezone: str = Field("UTC", description="Instrument timezone")

    # Trading sessions with start/end times
    sessions: Dict[str, Tuple[str, str]] = Field(
        default_factory=lambda: {
            "ASIAN": ("00:00", "09:00"),
            "LONDON": ("08:00", "17:00"),
            "NEW_YORK": ("13:00", "22:00"),
        },
        description="Trading sessions with start/end times in HH:MM format",
    )

    # Spread and volatility characteristics
    typical_spread: float = Field(0.0001, description="Typical spread in price units")
    avg_daily_range: float = Field(0.01, description="Average daily range in price units")

    @validator("sessions")
    def validate_sessions(cls, v: Dict[str, Tuple[str, str]]) -> Dict[str, Tuple[str, str]]:
        """Validate session time formats"""
        for session, (start, end) in v.items():
            try:
                time.fromisoformat(start)
                time.fromisoformat(end)
            except ValueError:
                raise ValueError(f"Invalid time format in session {session}: {start}-{end}")
        return v


class DimensionalReading(BaseModel):
    """
    Canonical output format for all dimensional sensors.
    Standardizes how each dimension communicates its perception to the orchestrator.
    """

    dimension: str = Field(..., description="Dimension name (WHY/HOW/WHAT/WHEN/ANOMALY)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Core signal components
    signal_strength: float = Field(..., ge=-1.0, le=1.0, description="Primary signal (-1 to +1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Signal confidence (0 to 1)")

    # Context and metadata
    regime: MarketRegime = Field(..., description="Detected market regime")
    context: Dict[str, Any] = Field(default_factory=dict, description="Dimension-specific context")

    # Quality metrics
    data_quality: float = Field(1.0, ge=0.0, le=1.0, description="Input data quality score")
    processing_time_ms: float = Field(0.0, description="Processing time in milliseconds")

    # Supporting evidence
    evidence: Dict[str, float] = Field(
        default_factory=dict, description="Supporting evidence scores"
    )
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")

    @validator("signal_strength")
    def validate_signal_strength(cls, v: float) -> float:
        """Ensure signal strength is within valid range"""
        if not -1.0 <= v <= 1.0:
            raise ValueError(f"Signal strength must be between -1.0 and +1.0, got {v}")
        return v

    @validator("confidence")
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is within valid range"""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return v


class MarketData(BaseModel):
    """
    Standardized market data structure for all dimensional engines.
    Replaces ad-hoc data structures with a canonical format.
    """

    symbol: str
    timestamp: datetime

    # OHLCV data
    open: float
    high: float
    low: float
    close: float
    volume: float

    # Bid/Ask data
    bid: float
    ask: float

    # Derived fields
    spread: float = Field(default=0.0)
    mid_price: float = Field(default=0.0)

    # Optional Level 2 data
    bid_volume: Optional[float] = None
    ask_volume: Optional[float] = None

    # Data source metadata
    source: str = Field("unknown", description="Data source identifier")
    latency_ms: float = Field(0.0, description="Data latency in milliseconds")

    @root_validator()
    def calculate_derived_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate derived fields after validation."""
        bid = values.get("bid")
        ask = values.get("ask")
        if bid is not None and ask is not None:
            values["spread"] = ask - bid
            values["mid_price"] = (bid + ask) / 2.0
        return values


class EconomicEvent(BaseModel):
    """
    Economic calendar event structure for the WHY dimension.
    """

    event_id: str
    timestamp: datetime
    currency: str
    event_name: str
    tier: EventTier

    # Event details
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None

    # Impact assessment
    impact_score: float = Field(0.0, ge=0.0, le=1.0)
    # (actual - forecast) / std_dev
    surprise_index: float = Field(0.0, ge=-1.0, le=1.0)

    # Time to event (for forward-looking analysis)
    time_to_event_hours: float = Field(0.0, description="Hours until event (negative if past)")


class DimensionalSensor(ABC):
    """
    Abstract base class for all dimensional sensors.
    Enforces consistent interface and behavior across all dimensions.
    """

    def __init__(self, instrument_meta: InstrumentMeta):
        self.instrument_meta = instrument_meta
        self.last_reading: Optional[DimensionalReading] = None
        self.is_initialized = False

    @abstractmethod
    async def update(self, market_data: MarketData) -> DimensionalReading:
        """
        Process new market data and return dimensional reading.

        Args:
            market_data: Latest market data

        Returns:
            DimensionalReading with current perception
        """

    @abstractmethod
    def snapshot(self) -> DimensionalReading:
        """
        Return current dimensional state without processing new data.

        Returns:
            Current DimensionalReading or default if not initialized
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset sensor state for new trading session or instrument."""

    def get_dimension_name(self) -> str:
        """Return the dimension name for this sensor."""
        return self.__class__.__name__.replace("Engine", "").replace("Sensor", "").upper()


class SystemHealth(BaseModel):
    """
    System health metrics for monitoring and diagnostics.
    """

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Performance metrics
    total_processing_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

    # Data quality metrics
    data_freshness_seconds: float = 0.0
    missing_data_count: int = 0
    error_count: int = 0

    # Dimensional health
    active_dimensions: List[str] = Field(default_factory=list)
    failed_dimensions: List[str] = Field(default_factory=list)

    # Overall system status
    overall_health_score: float = Field(1.0, ge=0.0, le=1.0)
    status: str = "healthy"  # healthy, degraded, critical

    def is_healthy(self) -> bool:
        """Check if system is in healthy state."""
        return (
            self.overall_health_score >= 0.7
            and len(self.failed_dimensions) == 0
            and self.error_count == 0
        )


# Type aliases for clarity
SignalStrength = float  # -1.0 to +1.0
Confidence = float  # 0.0 to 1.0
Price = float
Volume = float
Timestamp = datetime
