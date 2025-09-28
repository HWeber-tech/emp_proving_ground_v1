"""Generate a weekly capital efficiency memo comparing realised vs target budgets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Iterable

from src.config.risk.risk_config import RiskConfig
from src.trading.order_management import PositionTracker
from src.trading.order_management.journal_loader import (
    ProcessedFill,
    load_order_journal,
    replay_journal_into_tracker,
)


@dataclass(slots=True)
class DailyCapitalEfficiency:
    trading_day: date
    total_exposure: float
    target_exposure: float
    utilisation: float
    realised_pnl: float
    unrealised_pnl: float
    trades: int
    notional_traded: float


@dataclass(slots=True)
class CapitalEfficiencyMemo:
    days: list[DailyCapitalEfficiency]
    target_exposure: float
    risk_per_trade_budget: float
    total_notional_traded: float
    average_utilisation: float
    max_utilisation: float
    breach_days: list[DailyCapitalEfficiency]

    def render_markdown(self) -> str:
        if not self.days:
            return "# Capital Efficiency Memo\n\n_No journal fills detected for the selected period._"

        start = self.days[0].trading_day.isoformat()
        end = self.days[-1].trading_day.isoformat()

        lines = [
            "# Capital Efficiency Memo",
            f"_Period: {start} → {end}_",
            "",
            "## Executive Summary",
            f"* Target exposure budget: **{self.target_exposure:,.2f}**",
            f"* Risk-per-trade budget: **{self.risk_per_trade_budget:,.2f}**",
            f"* Total notional traded: **{self.total_notional_traded:,.2f}**",
            f"* Average utilisation: **{self.average_utilisation:.1%}**",
            f"* Peak utilisation: **{self.max_utilisation:.1%}**",
        ]

        if self.breach_days:
            breaches = ", ".join(day.trading_day.isoformat() for day in self.breach_days)
            lines.append(f"* ⚠️ Utilisation exceeded 100% on: **{breaches}**")
        else:
            lines.append("* ✅ No utilisation breaches detected")

        lines.extend(["", "## Daily Utilisation", "", _render_daily_table(self.days), ""])

        lines.append("## Observations")
        if self.breach_days:
            for day in self.breach_days:
                lines.append(
                    "- {} utilisation of {:.1%} exceeded the target exposure budget."
                    .format(day.trading_day.isoformat(), day.utilisation)
                )
        else:
            lines.append("- Exposure stayed within the configured budget for all sessions.")

        return "\n".join(lines)


def _render_daily_table(rows: Iterable[DailyCapitalEfficiency]) -> str:
    headers = [
        ("Date", 12),
        ("Exposure", 14),
        ("Target", 14),
        ("Utilisation", 12),
        ("Realised", 14),
        ("Unrealised", 14),
        ("Trades", 8),
        ("Turnover", 14),
    ]
    header_row = " ".join(title.ljust(width) for title, width in headers)
    divider = "-" * len(header_row)
    lines = [header_row, divider]

    for row in rows:
        lines.append(
            " ".join(
                [
                    row.trading_day.isoformat().ljust(12),
                    f"{row.total_exposure:,.2f}".rjust(14),
                    f"{row.target_exposure:,.2f}".rjust(14),
                    f"{row.utilisation:.0%}".rjust(12),
                    f"{row.realised_pnl:,.2f}".rjust(14),
                    f"{row.unrealised_pnl:,.2f}".rjust(14),
                    str(row.trades).rjust(8),
                    f"{row.notional_traded:,.2f}".rjust(14),
                ]
            )
        )

    return "\n".join(lines)


def compute_capital_efficiency(
    fills: Iterable[ProcessedFill],
    *,
    risk_config: RiskConfig,
    account_balance: float,
) -> CapitalEfficiencyMemo:
    if account_balance <= 0:
        raise ValueError("account_balance must be positive")

    target_exposure = account_balance * float(risk_config.max_total_exposure_pct)
    risk_per_trade_budget = account_balance * float(risk_config.max_risk_per_trade_pct)

    working_tracker = PositionTracker()
    totals = _DailyTotals(working_tracker)
    aggregations: dict[date, _DailyAccumulator] = defaultdict(_DailyAccumulator)

    for fill in fills:
        trading_day = fill.timestamp.date()
        daily = aggregations[trading_day]
        daily.trades += 1
        daily.notional_traded += fill.notional

        working_tracker.record_fill(
            symbol=fill.symbol,
            quantity=fill.quantity,
            price=fill.price,
            account=fill.account,
        )
        working_tracker.update_mark_price(fill.symbol, fill.price)

        totals.refresh()
        daily.total_exposure = totals.exposure
        daily.realised_pnl = totals.realised
        daily.unrealised_pnl = totals.unrealised

    days = []
    for trading_day in sorted(aggregations):
        snapshot = aggregations[trading_day]
        utilisation = (
            snapshot.total_exposure / target_exposure if target_exposure > 0 else 0.0
        )
        days.append(
            DailyCapitalEfficiency(
                trading_day=trading_day,
                total_exposure=snapshot.total_exposure,
                target_exposure=target_exposure,
                utilisation=utilisation,
                realised_pnl=snapshot.realised_pnl,
                unrealised_pnl=snapshot.unrealised_pnl,
                trades=snapshot.trades,
                notional_traded=snapshot.notional_traded,
            )
        )

    days.sort(key=lambda row: row.trading_day)

    total_notional = sum(day.notional_traded for day in days)
    average_utilisation = mean(day.utilisation for day in days) if days else 0.0
    max_utilisation = max((day.utilisation for day in days), default=0.0)
    breach_days = [day for day in days if day.utilisation > 1.0]

    return CapitalEfficiencyMemo(
        days=days,
        target_exposure=target_exposure,
        risk_per_trade_budget=risk_per_trade_budget,
        total_notional_traded=total_notional,
        average_utilisation=average_utilisation,
        max_utilisation=max_utilisation,
        breach_days=breach_days,
    )


class _DailyAccumulator:
    __slots__ = ("total_exposure", "realised_pnl", "unrealised_pnl", "trades", "notional_traded")

    def __init__(self) -> None:
        self.total_exposure = 0.0
        self.realised_pnl = 0.0
        self.unrealised_pnl = 0.0
        self.trades = 0
        self.notional_traded = 0.0


class _DailyTotals:
    __slots__ = ("tracker", "exposure", "realised", "unrealised")

    def __init__(self, tracker: PositionTracker) -> None:
        self.tracker = tracker
        self.exposure = 0.0
        self.realised = 0.0
        self.unrealised = 0.0

    def refresh(self) -> None:
        self.exposure = self.tracker.total_exposure()
        realised = 0.0
        unrealised = 0.0
        for snapshot in self.tracker.iter_positions():
            realised += snapshot.realized_pnl
            if snapshot.unrealized_pnl is not None:
                unrealised += snapshot.unrealized_pnl
        self.realised = realised
        self.unrealised = unrealised


def _load_risk_config(path: Path | None) -> RiskConfig:
    if path is None:
        return RiskConfig()

    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "YAML support requires PyYAML; install it or provide JSON",  # noqa: TRY003
            ) from exc
        data = yaml.safe_load(text)  # type: ignore[attr-defined]

    if not isinstance(data, dict):
        raise RuntimeError("Risk config file must contain a mapping")

    return RiskConfig(**data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "journal",
        type=Path,
        nargs="?",
        default=Path("data_foundation/events/order_events.parquet"),
        help="Path to the order event journal",
    )
    parser.add_argument(
        "--risk-config",
        type=Path,
        help="Optional path to a risk configuration JSON/YAML file",
    )
    parser.add_argument(
        "--account-balance",
        type=float,
        default=100_000.0,
        help="Baseline equity used when computing risk budgets",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the markdown memo",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    risk_config = _load_risk_config(args.risk_config)
    records = load_order_journal(args.journal)
    tracker = PositionTracker()

    fills: list[ProcessedFill] = []

    def _capture(fill: ProcessedFill, _: PositionTracker) -> None:
        fills.append(fill)

    replay_journal_into_tracker(records, tracker, on_fill=_capture)

    memo = compute_capital_efficiency(
        fills,
        risk_config=risk_config,
        account_balance=args.account_balance,
    )

    markdown = memo.render_markdown()
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

