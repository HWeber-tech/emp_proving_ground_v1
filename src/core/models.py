"""
Core Models for EMP System

Shared Pydantic models used across the system for type safety and data contracts.
"""

from datetime import datetime
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field


class InstrumentMeta(BaseModel):
    """
    Canonical instrument metadata for trading operations.
    Provides essential trading parameters and configuration.
    """
    
    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    pip_size: float = Field(..., description="Pip size for the instrument")
    lot_size: int = Field(default=100000, description="Standard lot size")
    commission: float = Field(default=0.0, description="Commission per trade")
    spread: float = Field(default=0.0001, description="Typical spread")
    timezone: str = Field(default="UTC", description="Instrument timezone")
    
    # Trading parameters
    min_lot: float = Field(default=0.01, description="Minimum lot size")
    max_lot: float = Field(default=100.0, description="Maximum lot size")
    leverage: int = Field(default=100, description="Maximum leverage")
    
    # Risk parameters
    stop_level: float = Field(default=0.0, description="Stop level distance")
    freeze_level: float = Field(default=0.0, description="Freeze level distance")
    
    class Config:
        frozen = True


class TokenData(BaseModel):
    """OAuth token data for API authentication."""
    
    access_token: str = Field(..., description="Access token")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiry in seconds")
    refresh_token: Optional[str] = Field(None, description="Refresh token")
    scope: Optional[str] = Field(None, description="Token scope")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def expires_at(self) -> float:
        """Calculate absolute expiry time as timestamp."""
        return self.created_at.timestamp() + self.expires_in
    
    @property
    def is_expired(self) -> bool:
        """Check if token is expired (with 1 hour buffer)."""
        import time
        return time.time() > (self.expires_at - 3600)


class SystemHealth(BaseModel):
    """System health metrics for monitoring."""
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field(default="healthy", description="Overall system status")
    uptime_seconds: float = Field(default=0.0, description="System uptime")
    
    # Component health
    components: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Component health status"
    )
    
    # Performance metrics
    memory_usage_mb: float = Field(default=0.0, description="Memory usage in MB")
    cpu_usage_percent: float = Field(default=0.0, description="CPU usage percentage")
    
    # Error tracking
    error_count: int = Field(default=0, description="Total error count")
    last_error: Optional[str] = Field(None, description="Last error message")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TradingConfig(BaseModel):
    """Trading configuration parameters."""
    
    symbol: str
    risk_per_trade: float = Field(default=0.01, description="Risk per trade as percentage")
    max_positions: int = Field(default=1, description="Maximum open positions")
    stop_loss_pips: int = Field(default=50, description="Stop loss in pips")
    take_profit_pips: int = Field(default=100, description="Take profit in pips")
    
    # Time-based restrictions
    trading_hours: Dict[str, str] = Field(
        default_factory=lambda: {
            "start": "00:00",
            "end": "23:59"
        }
    )
    
    # Risk management
    max_daily_loss: float = Field(default=0.05, description="Maximum daily loss")
    max_drawdown: float = Field(default=0.1, description="Maximum drawdown")
    
    class Config:
        frozen = True
