"""Return-on-investment telemetry helpers aligned with the roadmap's commercial goals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Mapping

from src.core.event_bus import Event, EventBus


class RoiStatus(StrEnum):
    """Execution status for ROI posture telemetry."""

    ahead = "ahead"
    tracking = "tracking"
    at_risk = "at_risk"


@dataclass(frozen=True)
class RoiCostModel:
    """Configuration describing capital, targets, and cost assumptions."""

    initial_capital: float
    target_annual_roi: float
    infrastructure_daily_cost: float
    broker_fee_flat: float = 0.0
    broker_fee_bps: float = 0.0

    @classmethod
    def bootstrap_defaults(cls, initial_capital: float) -> "RoiCostModel":
        """Zero-cost bootstrap deployment assumptions."""

        return cls(
            initial_capital=max(0.0, float(initial_capital)),
            target_annual_roi=0.20,
            infrastructure_daily_cost=0.0,
            broker_fee_flat=0.0,
            broker_fee_bps=0.0,
        )

    @classmethod
    def institutional_defaults(cls, initial_capital: float) -> "RoiCostModel":
        """Institutional tier defaults based on the concept blueprint."""

        monthly_cost = 250.0  # €250 monthly operating budget (concept promise)
        return cls(
            initial_capital=max(0.0, float(initial_capital)),
            target_annual_roi=0.25,
            infrastructure_daily_cost=monthly_cost / 30.0,
            broker_fee_flat=2.5,
            broker_fee_bps=0.5,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "initial_capital": float(self.initial_capital),
            "target_annual_roi": float(self.target_annual_roi),
            "infrastructure_daily_cost": float(self.infrastructure_daily_cost),
            "broker_fee_flat": float(self.broker_fee_flat),
            "broker_fee_bps": float(self.broker_fee_bps),
        }


@dataclass(frozen=True)
class RoiTelemetrySnapshot:
    """ROI posture snapshot emitted on ``telemetry.operational.roi``."""

    status: RoiStatus
    generated_at: datetime
    initial_capital: float
    current_equity: float
    gross_pnl: float
    net_pnl: float
    infrastructure_cost: float
    fees: float
    days_active: float
    executed_trades: int
    total_notional: float
    roi: float
    annualised_roi: float
    gross_roi: float
    gross_annualised_roi: float
    breakeven_daily_return: float
    target_annual_roi: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "status": self.status.value,
            "generated_at": self.generated_at.isoformat(),
            "initial_capital": self.initial_capital,
            "current_equity": self.current_equity,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "infrastructure_cost": self.infrastructure_cost,
            "fees": self.fees,
            "days_active": self.days_active,
            "executed_trades": self.executed_trades,
            "total_notional": self.total_notional,
            "roi": self.roi,
            "annualised_roi": self.annualised_roi,
            "gross_roi": self.gross_roi,
            "gross_annualised_roi": self.gross_annualised_roi,
            "breakeven_daily_return": self.breakeven_daily_return,
            "target_annual_roi": self.target_annual_roi,
        }


_MIN_DAY_FRACTION = 1.0 / 24.0  # Treat sub-hour windows conservatively.


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def evaluate_roi_posture(
    portfolio_state: Mapping[str, object],
    cost_model: RoiCostModel,
    *,
    executed_trades: int,
    total_notional: float,
    period_start: datetime,
    as_of: datetime | None = None,
) -> RoiTelemetrySnapshot:
    """Compute ROI telemetry using portfolio state and configured costs."""

    if as_of is None:
        as_of = datetime.now(tz=timezone.utc)

    elapsed = (as_of - period_start).total_seconds()
    days_active = max(elapsed / 86400.0, _MIN_DAY_FRACTION)

    equity = _safe_float(portfolio_state.get("equity"), cost_model.initial_capital)
    gross_pnl = _safe_float(portfolio_state.get("total_pnl"), equity - cost_model.initial_capital)

    fees_flat = float(executed_trades) * cost_model.broker_fee_flat
    fees_bps = float(total_notional) * (cost_model.broker_fee_bps / 10_000.0)
    fees = fees_flat + fees_bps

    infrastructure_cost = cost_model.infrastructure_daily_cost * days_active
    net_pnl = gross_pnl - fees - infrastructure_cost

    initial_capital = max(cost_model.initial_capital, 1e-9)
    roi = net_pnl / initial_capital
    gross_roi = gross_pnl / initial_capital

    scale = 365.25 / days_active
    annualised_roi = roi * scale
    gross_annualised_roi = gross_roi * scale

    if annualised_roi >= cost_model.target_annual_roi:
        status = RoiStatus.ahead
    elif annualised_roi >= cost_model.target_annual_roi * 0.5:
        status = RoiStatus.tracking
    else:
        status = RoiStatus.at_risk

    breakeven_daily_return = (
        cost_model.infrastructure_daily_cost + (fees / days_active)
    ) / initial_capital

    snapshot = RoiTelemetrySnapshot(
        status=status,
        generated_at=as_of,
        initial_capital=cost_model.initial_capital,
        current_equity=equity,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        infrastructure_cost=infrastructure_cost,
        fees=fees,
        days_active=days_active,
        executed_trades=int(executed_trades),
        total_notional=float(total_notional),
        roi=roi,
        annualised_roi=annualised_roi,
        gross_roi=gross_roi,
        gross_annualised_roi=gross_annualised_roi,
        breakeven_daily_return=breakeven_daily_return,
        target_annual_roi=cost_model.target_annual_roi,
    )
    return snapshot


def format_roi_markdown(snapshot: RoiTelemetrySnapshot) -> str:
    """Render a Markdown summary for dashboards and runbooks."""

    lines = [
        f"**Status:** {snapshot.status.value.upper()}",  # headline status
        "**Capital:** initial={:,.2f} equity={:,.2f}".format(
            snapshot.initial_capital, snapshot.current_equity
        ),
        "**PnL:** gross={:,.2f} net={:,.2f}".format(snapshot.gross_pnl, snapshot.net_pnl),
        "**ROI:** net={:.2%} annualised={:.2%} (target {:.2%})".format(
            snapshot.roi, snapshot.annualised_roi, snapshot.target_annual_roi
        ),
        "**Cost drag:** infra={:,.2f} fees={:,.2f} ({} trades)".format(
            snapshot.infrastructure_cost,
            snapshot.fees,
            snapshot.executed_trades,
        ),
        "**Breakeven daily return:** {:.2%}".format(snapshot.breakeven_daily_return),
    ]
    return "\n".join(lines)


async def publish_roi_snapshot(
    event_bus: EventBus,
    snapshot: RoiTelemetrySnapshot,
    *,
    source: str = "trading_manager",
) -> None:
    """Publish ROI telemetry on ``telemetry.operational.roi``."""

    payload = snapshot.as_dict()
    payload["markdown"] = format_roi_markdown(snapshot)
    event = Event(type="telemetry.operational.roi", payload=payload, source=source)
    await event_bus.publish(event)
