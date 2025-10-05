"""Telemetry helpers for recorded sensory replay evaluations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Mapping, MutableMapping, Sequence

from src.evolution.evaluation.recorded_replay import (
    RecordedEvaluationResult,
    RecordedTrade,
)
from src.sensory.lineage import SensorLineageRecord, build_lineage_record

__all__ = [
    "RecordedReplayTelemetrySnapshot",
    "summarise_recorded_replay",
]


def _coerce_parameters(parameters: Mapping[str, Any] | None) -> dict[str, float]:
    cleaned: MutableMapping[str, float] = {}
    if not parameters:
        return {}
    for key, value in parameters.items():
        try:
            cleaned[str(key)] = float(value)  # type: ignore[arg-type]
        except Exception:
            continue
    return dict(cleaned)


def _serialise_trade(trade: RecordedTrade | None) -> dict[str, Any] | None:
    if trade is None:
        return None
    payload = trade.as_dict()
    duration = (trade.closed_at - trade.opened_at).total_seconds() / 60.0
    payload["holding_minutes"] = round(duration, 4)
    return payload


def _profit_factor(trades: Sequence[RecordedTrade]) -> float:
    gains = sum(trade.return_pct for trade in trades if trade.return_pct > 0)
    losses = sum(trade.return_pct for trade in trades if trade.return_pct < 0)
    if gains == 0 and losses == 0:
        return 0.0
    if losses == 0:
        return float("inf")
    return gains / abs(losses)


def _exposure_minutes(trades: Sequence[RecordedTrade]) -> float:
    total_seconds = sum(
        (trade.closed_at - trade.opened_at).total_seconds() for trade in trades
    )
    return total_seconds / 60.0


def _rank_trade(
    trades: Sequence[RecordedTrade], *, reverse: bool
) -> RecordedTrade | None:
    if not trades:
        return None
    if reverse:
        return max(trades, key=lambda trade: trade.return_pct, default=None)
    return min(trades, key=lambda trade: trade.return_pct, default=None)


def _elevate_status(current: str, candidate: str) -> str:
    order = {"normal": 0, "warn": 1, "alert": 2}
    if order.get(candidate, 0) > order.get(current, 0):
        return candidate
    return current


@dataclass(frozen=True)
class RecordedReplayTelemetrySnapshot:
    """Aggregated telemetry describing a recorded replay evaluation."""

    generated_at: datetime
    status: str
    genome_id: str
    dataset_id: str | None
    evaluation_id: str | None
    metrics: Mapping[str, float]
    trade_summary: Mapping[str, Any]
    lineage: SensorLineageRecord
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "generated_at": self.generated_at.isoformat(),
            "status": self.status,
            "genome_id": self.genome_id,
            "metrics": dict(self.metrics),
            "trade_summary": dict(self.trade_summary),
            "lineage": self.lineage.as_dict(),
        }
        if self.dataset_id is not None:
            payload["dataset_id"] = self.dataset_id
        if self.evaluation_id is not None:
            payload["evaluation_id"] = self.evaluation_id
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def to_markdown(self) -> str:
        lines = [
            f"### Recorded replay telemetry for genome `{self.genome_id}`",
            "",
            "| Metric | Value |",
            "| --- | --- |",
        ]
        total_return = self.metrics.get("total_return")
        max_drawdown = self.metrics.get("max_drawdown")
        sharpe = self.metrics.get("sharpe_ratio")
        win_rate = self.metrics.get("win_rate")
        volatility = self.metrics.get("volatility")
        trades = self.metrics.get("trades")
        exposure = self.trade_summary.get("exposure_minutes")
        profit_factor = self.trade_summary.get("profit_factor")

        if total_return is not None:
            lines.append(f"| Total return | {total_return:.2%} |")
        if max_drawdown is not None:
            lines.append(f"| Max drawdown | {max_drawdown:.2%} |")
        if sharpe is not None:
            lines.append(f"| Sharpe ratio | {sharpe:.3f} |")
        if volatility is not None:
            lines.append(f"| Volatility | {volatility:.3f} |")
        if win_rate is not None:
            lines.append(f"| Win rate | {win_rate:.2%} |")
        if trades is not None:
            lines.append(f"| Trades | {int(trades)} |")
        if exposure is not None:
            lines.append(f"| Exposure (minutes) | {exposure:.2f} |")
        if profit_factor is not None:
            if profit_factor == float("inf"):
                display = "∞"
            else:
                display = f"{profit_factor:.3f}"
            lines.append(f"| Profit factor | {display} |")

        best = self.trade_summary.get("best_trade")
        worst = self.trade_summary.get("worst_trade")
        if isinstance(best, dict) and best:
            lines.append(
                f"| Best trade | {best.get('return_pct', 0.0):.2%} ({best.get('holding_minutes', 0.0):.1f} min) |"
            )
        if isinstance(worst, dict) and worst:
            lines.append(
                f"| Worst trade | {worst.get('return_pct', 0.0):.2%} ({worst.get('holding_minutes', 0.0):.1f} min) |"
            )

        return "\n".join(lines)


def summarise_recorded_replay(
    result: RecordedEvaluationResult,
    *,
    genome_id: str,
    dataset_id: str | None = None,
    evaluation_id: str | None = None,
    parameters: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    warn_drawdown: float = 0.15,
    alert_drawdown: float = 0.25,
) -> RecordedReplayTelemetrySnapshot:
    """Fuse replay evaluation metrics into a telemetry snapshot."""

    parameters_payload = _coerce_parameters(parameters)
    trades: Sequence[RecordedTrade] = result.trade_log

    best_trade = _rank_trade(trades, reverse=True)
    worst_trade = _rank_trade(trades, reverse=False)
    profit_factor = _profit_factor(trades)
    exposure = _exposure_minutes(trades)

    metrics: dict[str, float] = {
        "total_return": float(result.total_return),
        "max_drawdown": float(result.max_drawdown),
        "sharpe_ratio": float(result.sharpe_ratio),
        "volatility": float(result.volatility),
        "win_rate": float(result.win_rate),
        "trades": float(result.trades),
        "average_trade_duration_minutes": float(result.average_trade_duration_minutes),
    }

    trade_summary: dict[str, Any] = {
        "profit_factor": profit_factor,
        "exposure_minutes": exposure,
    }
    serialised_best = _serialise_trade(best_trade)
    serialised_worst = _serialise_trade(worst_trade)
    if serialised_best is not None:
        trade_summary["best_trade"] = serialised_best
    if serialised_worst is not None:
        trade_summary["worst_trade"] = serialised_worst

    status = "normal"
    if result.max_drawdown >= warn_drawdown:
        status = _elevate_status(status, "warn")
    if result.max_drawdown >= alert_drawdown:
        status = _elevate_status(status, "alert")
    if result.total_return <= 0:
        status = _elevate_status(status, "alert")
    elif result.total_return < 0.02:
        status = _elevate_status(status, "warn")
    if result.trades > 0 and result.win_rate < 0.35:
        status = _elevate_status(status, "warn")
    if result.trades == 0:
        status = _elevate_status(status, "warn")

    lineage_inputs: dict[str, Any] = {"parameters": parameters_payload}
    if dataset_id is not None:
        lineage_inputs["dataset_id"] = dataset_id
    if evaluation_id is not None:
        lineage_inputs["evaluation_id"] = evaluation_id

    lineage_outputs: dict[str, Any] = {
        "total_return": metrics["total_return"],
        "max_drawdown": metrics["max_drawdown"],
        "win_rate": metrics["win_rate"],
        "trades": int(result.trades),
        "status": status,
    }

    lineage = build_lineage_record(
        "EVOLUTION",
        "evolution.recorded_replay",
        inputs=lineage_inputs,
        outputs=lineage_outputs,
        telemetry={
            "profit_factor": profit_factor,
            "exposure_minutes": exposure,
            "sharpe_ratio": metrics["sharpe_ratio"],
        },
        metadata={
            "genome_id": genome_id,
            "drawdown_thresholds": {
                "warn": warn_drawdown,
                "alert": alert_drawdown,
            },
        },
    )

    snapshot_metadata: dict[str, Any] = dict(metadata or {})
    if parameters_payload:
        snapshot_metadata.setdefault("parameters", parameters_payload)

    return RecordedReplayTelemetrySnapshot(
        generated_at=datetime.now(tz=UTC),
        status=status,
        genome_id=genome_id,
        dataset_id=dataset_id,
        evaluation_id=evaluation_id,
        metrics=metrics,
        trade_summary=trade_summary,
        lineage=lineage,
        metadata=snapshot_metadata,
    )
