"""ICT-style microstructure signal adapters for strategies.

This module bridges the institutional footprint detectors in
:mod:`src.sensory.organs.dimensions.institutional_tracker` with the
strategy layer.  It converts generic market data payloads into
``MarketData`` records, executes the footprint analysis, and reduces the
results into light-weight features that can be consumed by strategies
without leaking implementation details.

The implementation targets the unchecked roadmap item from
``docs/High-Impact Development Roadmap.md`` that requires integrating
fair value gap and liquidity sweep analytics into strategies once
validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

from src.core.base import MarketData
from src.operational.structured_logging import get_logger
from src.sensory.organs.dimensions.institutional_tracker import (
    InstitutionalFootprint,
    InstitutionalFootprintHunter,
)

__all__ = [
    "ICTMicrostructureFeatures",
    "ICTMicrostructureAnalyzer",
]

logger = get_logger(__name__)


def _clamp(value: float, *, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(slots=True)
class ICTMicrostructureFeatures:
    """Summary of microstructure signals exposed to strategies."""

    institutional_bias: str | None
    smart_money_flow: float
    fair_value_gap_count: int
    recent_fvg_type: str | None
    strongest_fvg_strength: float | None
    liquidity_sweep_count: int
    recent_liquidity_sweep_direction: str | None
    liquidity_sweep_bias: str | None
    confidence: float
    key_levels: tuple[float, ...] = field(default_factory=tuple)
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "confidence",
            float(_clamp(float(self.confidence), lower=0.0, upper=1.0)),
        )
        object.__setattr__(
            self,
            "smart_money_flow",
            float(_clamp(float(self.smart_money_flow), lower=-1.0, upper=1.0)),
        )
        if not isinstance(self.key_levels, tuple):
            object.__setattr__(self, "key_levels", tuple(self.key_levels))

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-serialisable metadata."""

        metadata: dict[str, Any] = {
            "institutional_bias": self.institutional_bias,
            "smart_money_flow": self.smart_money_flow,
            "fair_value_gap_count": self.fair_value_gap_count,
            "recent_fvg_type": self.recent_fvg_type,
            "strongest_fvg_strength": self.strongest_fvg_strength,
            "liquidity_sweep_count": self.liquidity_sweep_count,
            "recent_liquidity_sweep_direction": self.recent_liquidity_sweep_direction,
            "liquidity_sweep_bias": self.liquidity_sweep_bias,
            "confidence": self.confidence,
            "key_levels": list(self.key_levels),
        }
        metadata.update(self.extras)
        return metadata

    def alignment_assessment(self, action: str) -> tuple[float, dict[str, float]]:
        """Compute directional alignment score and component breakdown.

        Parameters
        ----------
        action:
            Strategy action (``"BUY"`` or ``"SELL"``). Any other action
            yields a neutral (0.0) alignment.
        """

        action_upper = action.upper()
        if action_upper not in {"BUY", "SELL"}:
            return (0.0, {})

        direction = 1.0 if action_upper == "BUY" else -1.0
        components: dict[str, float] = {}

        bias_weight = 0.35
        fvg_weight = 0.25
        sweep_weight = 0.2
        flow_weight = 0.2

        if self.institutional_bias == "bullish":
            components["institutional_bias"] = direction * bias_weight
        elif self.institutional_bias == "bearish":
            components["institutional_bias"] = -direction * bias_weight
        else:
            components["institutional_bias"] = 0.0

        if self.recent_fvg_type == "bullish" and self.fair_value_gap_count > 0:
            components["fair_value_gap"] = direction * fvg_weight
        elif self.recent_fvg_type == "bearish" and self.fair_value_gap_count > 0:
            components["fair_value_gap"] = -direction * fvg_weight
        else:
            components["fair_value_gap"] = 0.0

        if self.liquidity_sweep_bias == "bullish" and self.liquidity_sweep_count > 0:
            components["liquidity_sweep"] = direction * sweep_weight
        elif self.liquidity_sweep_bias == "bearish" and self.liquidity_sweep_count > 0:
            components["liquidity_sweep"] = -direction * sweep_weight
        else:
            components["liquidity_sweep"] = 0.0

        components["smart_money_flow"] = direction * flow_weight * self.smart_money_flow

        score = _clamp(sum(components.values()), lower=-1.0, upper=1.0)
        return (score, components)

    @classmethod
    def from_footprint(cls, footprint: InstitutionalFootprint) -> "ICTMicrostructureFeatures":
        fvgs = list(footprint.fair_value_gaps)
        sweeps = list(footprint.liquidity_sweeps)

        extras: dict[str, Any] = {}

        recent_fvg_type: str | None = None
        strongest_fvg_strength: float | None = None
        if fvgs:
            fvgs_sorted = sorted(fvgs, key=lambda gap: gap.timestamp)
            recent_gap = fvgs_sorted[-1]
            recent_fvg_type = recent_gap.type
            strongest_gap = max(fvgs, key=lambda gap: gap.strength)
            strongest_fvg_strength = float(strongest_gap.strength)
            extras["recent_fair_value_gap"] = {
                "type": recent_gap.type,
                "start_price": float(recent_gap.start_price),
                "end_price": float(recent_gap.end_price),
                "strength": float(recent_gap.strength),
                "fill_probability": float(recent_gap.fill_probability),
                "imbalance_ratio": float(recent_gap.imbalance_ratio),
                "timestamp": recent_gap.timestamp.replace(tzinfo=timezone.utc).isoformat(),
            }
            extras["strongest_fair_value_gap"] = {
                "type": strongest_gap.type,
                "strength": float(strongest_gap.strength),
                "imbalance_ratio": float(strongest_gap.imbalance_ratio),
            }
        else:
            extras["recent_fair_value_gap"] = None
            extras["strongest_fair_value_gap"] = None

        recent_liquidity_sweep_direction: str | None = None
        liquidity_sweep_bias: str | None = None
        if sweeps:
            recent_sweep = sweeps[-1]
            recent_liquidity_sweep_direction = recent_sweep.direction
            liquidity_sweep_bias = "bullish" if recent_sweep.direction == "down" else "bearish"
            extras["recent_liquidity_sweep"] = {
                "direction": recent_sweep.direction,
                "sweep_level": float(recent_sweep.sweep_level),
                "liquidity_pool": recent_sweep.liquidity_pool,
                "sweep_size": float(recent_sweep.sweep_size),
                "volume_spike": float(recent_sweep.volume_spike),
                "reversal_probability": float(recent_sweep.reversal_probability),
                "institutional_follow_through": bool(recent_sweep.institutional_follow_through),
            }
        else:
            extras["recent_liquidity_sweep"] = None

        extras["key_levels"] = [float(level) for level in footprint.key_levels]

        return cls(
            institutional_bias=footprint.institutional_bias,
            smart_money_flow=float(footprint.smart_money_flow),
            fair_value_gap_count=len(fvgs),
            recent_fvg_type=recent_fvg_type,
            strongest_fvg_strength=strongest_fvg_strength,
            liquidity_sweep_count=len(sweeps),
            recent_liquidity_sweep_direction=recent_liquidity_sweep_direction,
            liquidity_sweep_bias=liquidity_sweep_bias,
            confidence=float(footprint.confidence_score),
            key_levels=tuple(float(level) for level in footprint.key_levels),
            extras=extras,
        )


class ICTMicrostructureAnalyzer:
    """Adapter that produces :class:`ICTMicrostructureFeatures` for strategies."""

    def __init__(
        self,
        *,
        footprint_hunter: InstitutionalFootprintHunter | None = None,
        minimum_candles: int = 10,
    ) -> None:
        self._footprint_hunter = footprint_hunter or InstitutionalFootprintHunter()
        self._minimum_candles = max(3, int(minimum_candles))

    async def summarise(
        self, market_data: Mapping[str, Any], symbol: str
    ) -> ICTMicrostructureFeatures | None:
        payload = market_data.get(symbol)
        if not isinstance(payload, Mapping):
            return None

        candles = self._normalise_payload(symbol, payload)
        if len(candles) < self._minimum_candles:
            return None

        try:
            footprint = await self._footprint_hunter.analyze_institutional_footprint(
                candles, symbol
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("microstructure_footprint_failed", symbol=symbol)
            return None

        return ICTMicrostructureFeatures.from_footprint(footprint)

    def _normalise_payload(
        self, symbol: str, payload: Mapping[str, Any]
    ) -> list[MarketData]:
        closes = self._extract_numeric_sequence(payload.get("close"))
        length = len(closes)
        if length == 0:
            return []

        opens = self._prepare_sequence(payload.get("open"), default=closes, length=length)
        highs = self._prepare_sequence(
            payload.get("high"),
            default=[max(o, c) for o, c in zip(opens, closes)],
            length=length,
        )
        lows = self._prepare_sequence(
            payload.get("low"),
            default=[min(o, c) for o, c in zip(opens, closes)],
            length=length,
        )
        volumes = self._prepare_sequence(
            payload.get("volume"), default=[0.0] * length, length=length
        )

        timestamps = self._prepare_timestamps(payload.get("timestamp"), length=length)

        candles: list[MarketData] = []
        for idx in range(length):
            candles.append(
                MarketData(
                    symbol=symbol,
                    timestamp=timestamps[idx],
                    open=opens[idx],
                    high=highs[idx],
                    low=lows[idx],
                    close=closes[idx],
                    volume=volumes[idx],
                    bid=closes[idx],
                    ask=closes[idx],
                    spread=0.0,
                    mid_price=closes[idx],
                )
            )

        return candles

    def _extract_numeric_sequence(self, value: object) -> list[float]:
        if not self._is_sequence(value):
            return []
        result = []
        for item in value:  # type: ignore[assignment]
            try:
                result.append(float(item))
            except (TypeError, ValueError):
                continue
        return result

    def _prepare_sequence(
        self, value: object, *, default: Iterable[float], length: int
    ) -> list[float]:
        if not self._is_sequence(value):
            base = list(default)
        else:
            base = []
            for item in value:  # type: ignore[assignment]
                try:
                    base.append(float(item))
                except (TypeError, ValueError):
                    base.append(float(base[-1] if base else 0.0))
        if len(base) < length:
            base.extend([float(base[-1] if base else 0.0)] * (length - len(base)))
        elif len(base) > length:
            base = base[-length:]
        return base

    def _prepare_timestamps(self, value: object, *, length: int) -> list[datetime]:
        if self._is_sequence(value):
            result: list[datetime] = []
            for item in value:  # type: ignore[assignment]
                ts = self._coerce_timestamp(item)
                result.append(ts)
            if len(result) >= length:
                return result[-length:]
            if result:
                last = result[-1]
            else:
                last = datetime.now(tz=timezone.utc)
            # Pad forward by one minute increments
            while len(result) < length:
                last = last + timedelta(minutes=1)
                result.append(last)
            return result
        # Generate synthetic timeline ending now
        end = datetime.now(tz=timezone.utc)
        return [end - timedelta(minutes=length - idx) for idx in range(length, 0, -1)]

    def _coerce_timestamp(self, value: object) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                pass
        return datetime.now(tz=timezone.utc)

    @staticmethod
    def _is_sequence(value: object) -> bool:
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes))
