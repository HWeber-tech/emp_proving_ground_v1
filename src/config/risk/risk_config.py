from __future__ import annotations

from decimal import Decimal
from typing import Dict

from pydantic import BaseModel, Field, root_validator, validator


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
    target_volatility_pct: Decimal = Field(
        default=Decimal("0.10"),
        description="Target volatility for volatility-target sizing",
    )
    volatility_window: int = Field(
        default=20,
        ge=1,
        description="Lookback window for realised volatility estimation",
    )
    max_volatility_leverage: Decimal = Field(
        default=Decimal("3.0"),
        description="Maximum leverage when applying volatility targeting",
    )
    volatility_annualisation_factor: Decimal = Field(
        default=Decimal("1.0"),
        description="Annualisation factor applied to realised volatility",
    )
    instrument_sector_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of instrument symbol to sector or asset class identifier",
    )
    sector_exposure_limits: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Maximum fraction of equity allocatable to each sector or asset class",
    )

    @validator(
        "max_risk_per_trade_pct",
        "max_total_exposure_pct",
        "max_drawdown_pct",
        "target_volatility_pct",
    )
    def validate_percentages(cls, v: Decimal) -> Decimal:
        if v <= 0 or v > 1:
            raise ValueError("Percentage values must be between 0 and 1")
        return v

    @validator("max_leverage", "max_volatility_leverage")
    def validate_leverage(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("Leverage must be positive")
        return v

    @validator("volatility_annualisation_factor")
    def validate_annualisation(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("Annualisation factor must be positive")
        return v

    @validator("min_position_size", "max_position_size")
    def validate_position_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Position sizes must be positive")
        return v

    @validator("sector_exposure_limits", each_item=True)
    def validate_sector_limits(cls, v: Decimal) -> Decimal:
        if v <= 0 or v > 1:
            raise ValueError("Sector exposure limits must be between 0 and 1")
        return v

    @root_validator
    def validate_consistency(cls, values: dict[str, object]) -> dict[str, object]:
        min_size = values.get("min_position_size")
        max_size = values.get("max_position_size")
        if isinstance(min_size, int) and isinstance(max_size, int) and min_size > max_size:
            raise ValueError("min_position_size cannot exceed max_position_size")

        max_risk = values.get("max_risk_per_trade_pct")
        max_exposure = values.get("max_total_exposure_pct")
        if isinstance(max_risk, Decimal) and isinstance(max_exposure, Decimal):
            if max_risk > max_exposure:
                raise ValueError("max_total_exposure_pct must be >= max_risk_per_trade_pct")

        mandatory_stop_loss = values.get("mandatory_stop_loss")
        research_mode = values.get("research_mode")
        if mandatory_stop_loss is False and research_mode is not True:
            raise ValueError("Disabling mandatory_stop_loss requires research_mode=True")

        return values


__all__ = ["RiskConfig"]
