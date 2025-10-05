from __future__ import annotations

import warnings
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

    @validator("sector_exposure_limits", pre=True)
    def _normalise_sector_limit_keys(
        cls, value: Dict[str, Decimal] | None
    ) -> Dict[str, Decimal]:
        if value is None:
            return {}

        if not isinstance(value, dict):
            raise TypeError("sector_exposure_limits must be provided as a mapping")

        normalised: Dict[str, Decimal] = {}
        for raw_sector, raw_limit in value.items():
            if raw_sector is None or str(raw_sector).strip() == "":
                raise ValueError("Sector names must be non-empty strings")

            sector_key = str(raw_sector).upper()
            if sector_key in normalised:
                raise ValueError(
                    "Duplicate sector exposure limit defined for sector %s" % sector_key
                )

            normalised[sector_key] = raw_limit

        return normalised

    @validator("sector_exposure_limits", each_item=True)
    def validate_sector_limits(cls, v: Decimal) -> Decimal:
        if v <= 0 or v > 1:
            raise ValueError("Sector exposure limits must be between 0 and 1")
        return v

    @validator("instrument_sector_map", pre=True)
    def _normalise_instrument_sector_map(
        cls, value: Dict[str, str] | None
    ) -> Dict[str, str]:
        if value is None:
            return {}

        if not isinstance(value, dict):
            raise TypeError("instrument_sector_map must be provided as a mapping")

        normalised: Dict[str, str] = {}
        for raw_symbol, raw_sector in value.items():
            if raw_symbol is None or str(raw_symbol).strip() == "":
                raise ValueError("Instrument symbols must be non-empty strings")

            if raw_sector is None or str(raw_sector).strip() == "":
                raise ValueError("Instrument sector identifiers must be non-empty strings")

            symbol_key = str(raw_symbol).upper()
            sector_key = str(raw_sector).upper()

            previous_sector = normalised.get(symbol_key)
            if previous_sector is not None and previous_sector != sector_key:
                raise ValueError(
                    f"Instrument {symbol_key} assigned to multiple sectors"
                )

            normalised[symbol_key] = sector_key

        return normalised

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

        max_drawdown = values.get("max_drawdown_pct")
        if isinstance(max_drawdown, Decimal) and isinstance(max_risk, Decimal):
            if max_drawdown < max_risk:
                raise ValueError("max_drawdown_pct must be >= max_risk_per_trade_pct")

        mandatory_stop_loss = values.get("mandatory_stop_loss")
        research_mode = values.get("research_mode")
        if mandatory_stop_loss is False and research_mode is not True:
            warnings.warn(
                "RiskConfig configured with mandatory_stop_loss=False outside research mode; "
                "ensure governance approval before running in production.",
                UserWarning,
                stacklevel=2,
            )

        sector_limits = values.get("sector_exposure_limits") or {}
        instrument_map = values.get("instrument_sector_map") or {}
        if instrument_map:
            missing_sectors = {
                sector for sector in instrument_map.values() if sector not in sector_limits
            }
            if missing_sectors:
                raise ValueError(
                    "Sector exposure limits required for sectors: %s"
                    % ", ".join(sorted(missing_sectors))
                )

        max_exposure_pct = values.get("max_total_exposure_pct")
        if isinstance(max_exposure_pct, Decimal):
            sector_total = Decimal("0")
            for sector, limit in sector_limits.items():
                if isinstance(limit, Decimal) and limit > max_exposure_pct:
                    raise ValueError(
                        "Sector %s exposure limit exceeds max_total_exposure_pct" % sector
                    )
                if isinstance(limit, Decimal):
                    sector_total += limit

            if sector_limits and sector_total > max_exposure_pct:
                raise ValueError(
                    "Combined sector_exposure_limits exceed max_total_exposure_pct"
                )

        return values


__all__ = ["RiskConfig"]
