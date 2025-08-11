from __future__ import annotations

from decimal import Decimal
from pydantic import BaseModel, Field, validator


class RiskConfig(BaseModel):
    """Configuration for risk management (canonical)."""

    max_risk_per_trade_pct: Decimal = Field(
        default=Decimal("0.02"),
        description="Maximum risk per trade as percentage of equity",
    )
    max_leverage: Decimal = Field(
        default=Decimal("10.0"),
        description="Maximum allowed leverage",
    )
    max_total_exposure_pct: Decimal = Field(
        default=Decimal("0.5"),
        description="Maximum total exposure as percentage of equity",
    )
    max_drawdown_pct: Decimal = Field(
        default=Decimal("0.25"),
        description="Maximum drawdown before stopping",
    )
    min_position_size: int = Field(
        default=1000,
        description="Minimum position size in units",
    )
    max_position_size: int = Field(
        default=1000000,
        description="Maximum position size in units",
    )
    mandatory_stop_loss: bool = Field(
        default=True,
        description="Whether stop loss is mandatory",
    )
    research_mode: bool = Field(
        default=False,
        description="Research mode disables some safety checks",
    )

    @validator("max_risk_per_trade_pct", "max_total_exposure_pct", "max_drawdown_pct")
    def validate_percentages(cls, v: Decimal) -> Decimal:
        if v <= 0 or v > 1:
            raise ValueError("Percentage values must be between 0 and 1")
        return v

    @validator("max_leverage")
    def validate_leverage(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("Leverage must be positive")
        return v


__all__ = ["RiskConfig"]