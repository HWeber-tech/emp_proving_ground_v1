"""Strategy-level KPI tracker for understanding and trading loops."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from statistics import fmean
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from src.operations.roi import RoiCostModel, RoiTelemetrySnapshot, evaluate_roi_posture


@dataclass(slots=True)
class _MetricAggregate:
    """Helper that tracks running totals for a numeric metric."""

    total: float = 0.0
    samples: int = 0

    def add(self, value: float) -> None:
        self.total += value
        self.samples += 1

    def mean(self) -> float | None:
        if not self.samples:
            return None
        return self.total / self.samples


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    """Best-effort float coercion that tolerates string values."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _coerce_ratio(value: Any) -> float | None:
    """Convert percentage/bps/pip strings into fractional returns."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None
        if text.endswith("%"):
            numeric = _coerce_float(text[:-1])
            return numeric / 100.0 if numeric is not None else None
        if text.endswith("bps"):
            numeric = _coerce_float(text[:-3])
            return numeric / 10_000.0 if numeric is not None else None
        if text.endswith("bp"):
            numeric = _coerce_float(text[:-2])
            return numeric / 10_000.0 if numeric is not None else None
        if text.endswith("pip"):
            numeric = _coerce_float(text[:-3])
            return numeric / 10_4 if numeric is not None else None
        if text.endswith("pips"):
            numeric = _coerce_float(text[:-4])
            return numeric / 10_4 if numeric is not None else None
        return _coerce_float(text)
    return None


@dataclass
class _FastWeightTelemetryAccumulator:
    """Aggregate sparsity metrics captured from fast-weight controllers."""

    samples: int = 0
    total_metric: _MetricAggregate = field(default_factory=_MetricAggregate)
    active_metric: _MetricAggregate = field(default_factory=_MetricAggregate)
    dormant_metric: _MetricAggregate = field(default_factory=_MetricAggregate)
    inhibitory_metric: _MetricAggregate = field(default_factory=_MetricAggregate)
    suppressed_inhibitory_metric: _MetricAggregate = field(default_factory=_MetricAggregate)
    active_percentage_metric: _MetricAggregate = field(default_factory=_MetricAggregate)
    sparsity_metric: _MetricAggregate = field(default_factory=_MetricAggregate)
    max_multiplier: float | None = None
    min_multiplier: float | None = None

    def add_from_mapping(self, metrics: Mapping[str, Any]) -> None:
        if not metrics:
            return

        recorded = False

        def _ingest(target: _MetricAggregate, key: str) -> None:
            nonlocal recorded
            value = _as_float(metrics.get(key))
            if value is None:
                return
            target.add(value)
            recorded = True

        _ingest(self.total_metric, "total")
        _ingest(self.active_metric, "active")
        _ingest(self.dormant_metric, "dormant")
        _ingest(self.inhibitory_metric, "inhibitory")
        _ingest(self.suppressed_inhibitory_metric, "suppressed_inhibitory")
        _ingest(self.active_percentage_metric, "active_percentage")
        _ingest(self.sparsity_metric, "sparsity")

        max_multiplier = _as_float(metrics.get("max_multiplier"))
        if max_multiplier is not None:
            self.max_multiplier = (
                max_multiplier if self.max_multiplier is None else max(self.max_multiplier, max_multiplier)
            )
            recorded = True

        min_multiplier = _as_float(metrics.get("min_multiplier"))
        if min_multiplier is not None:
            self.min_multiplier = (
                min_multiplier if self.min_multiplier is None else min(self.min_multiplier, min_multiplier)
            )
            recorded = True

        if recorded:
            self.samples += 1

    def summary(self) -> Mapping[str, Any]:
        if not self.samples:
            return {}

        def _maybe_mean(aggregate: _MetricAggregate, key: str) -> tuple[str, float] | None:
            value = aggregate.mean()
            if value is None:
                return None
            return key, value

        payload: dict[str, Any] = {"samples": self.samples}

        for aggregate, label in (
            (self.total_metric, "total_mean"),
            (self.active_metric, "active_mean"),
            (self.dormant_metric, "dormant_mean"),
            (self.inhibitory_metric, "inhibitory_mean"),
            (self.suppressed_inhibitory_metric, "suppressed_inhibitory_mean"),
            (self.active_percentage_metric, "active_percentage_mean"),
            (self.sparsity_metric, "sparsity_mean"),
        ):
            entry = _maybe_mean(aggregate, label)
            if entry is not None:
                key, value = entry
                payload[key] = value

        if self.max_multiplier is not None:
            payload["max_multiplier"] = self.max_multiplier
        if self.min_multiplier is not None:
            payload["min_multiplier"] = self.min_multiplier

        return payload


@dataclass
class _HebbianAdapterAccumulator:
    """Aggregate Hebbian multiplier telemetry per fast-weight adapter."""

    adapter_id: str
    samples: int = 0
    current_multiplier: _MetricAggregate = field(default_factory=_MetricAggregate)
    previous_multiplier: _MetricAggregate = field(default_factory=_MetricAggregate)
    delta: _MetricAggregate = field(default_factory=_MetricAggregate)
    feature_value: _MetricAggregate = field(default_factory=_MetricAggregate)
    last_multiplier: float | None = None
    max_multiplier: float | None = None
    min_multiplier: float | None = None

    def add(self, summary: Mapping[str, Any]) -> None:
        current = _as_float(summary.get("current_multiplier"))
        if current is None:
            return

        self.samples += 1
        self.current_multiplier.add(current)
        self.last_multiplier = current
        self.max_multiplier = current if self.max_multiplier is None else max(self.max_multiplier, current)
        self.min_multiplier = current if self.min_multiplier is None else min(self.min_multiplier, current)

        previous = _as_float(summary.get("previous_multiplier"))
        if previous is not None:
            self.previous_multiplier.add(previous)
            self.delta.add(current - previous)

        feature = _as_float(summary.get("feature_value"))
        if feature is not None:
            self.feature_value.add(feature)

    def summary(self) -> Mapping[str, Any]:
        if not self.samples:
            return {}

        payload: dict[str, Any] = {"samples": self.samples}

        current_mean = self.current_multiplier.mean()
        if current_mean is not None:
            payload["current_multiplier_mean"] = current_mean

        previous_mean = self.previous_multiplier.mean()
        if previous_mean is not None:
            payload["previous_multiplier_mean"] = previous_mean

        delta_mean = self.delta.mean()
        if delta_mean is not None:
            payload["delta_mean"] = delta_mean

        feature_mean = self.feature_value.mean()
        if feature_mean is not None:
            payload["feature_value_mean"] = feature_mean

        if self.max_multiplier is not None:
            payload["max_multiplier"] = self.max_multiplier
        if self.min_multiplier is not None:
            payload["min_multiplier"] = self.min_multiplier
        if self.last_multiplier is not None:
            payload["last_multiplier"] = self.last_multiplier

        return payload


@dataclass(frozen=True)
class StrategyModeSummary:
    """Aggregated metrics for a specific fast-weight mode."""

    trades: int
    pnl: float
    notional: float
    roi: float
    average_return: float | None

    def as_dict(self) -> dict[str, float | int | None]:
        payload: dict[str, float | int | None] = {
            "trades": self.trades,
            "pnl": self.pnl,
            "notional": self.notional,
            "roi": self.roi,
        }
        if self.average_return is not None:
            payload["average_return"] = self.average_return
        return payload


@dataclass(frozen=True)
class StrategyFastWeightBreakdown:
    """Comparison of fast-weight enabled vs disabled performance."""

    enabled: StrategyModeSummary | None
    disabled: StrategyModeSummary | None
    roi_uplift: float | None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.enabled is not None:
            payload["enabled"] = self.enabled.as_dict()
        if self.disabled is not None:
            payload["disabled"] = self.disabled.as_dict()
        if self.roi_uplift is not None:
            payload["roi_uplift"] = self.roi_uplift
        return payload


@dataclass(frozen=True)
class StrategyKpi:
    """Per-strategy KPIs used in the daily performance report."""

    strategy_id: str
    trades: int
    wins: int
    losses: int
    total_pnl: float
    total_notional: float
    roi: float
    win_rate: float
    average_return: float | None
    max_drawdown: float
    max_drawdown_pct: float | None
    fast_weight_breakdown: StrategyFastWeightBreakdown | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "strategy_id": self.strategy_id,
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "total_pnl": self.total_pnl,
            "total_notional": self.total_notional,
            "roi": self.roi,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
        }
        if self.average_return is not None:
            payload["average_return"] = self.average_return
        if self.max_drawdown_pct is not None:
            payload["max_drawdown_pct"] = self.max_drawdown_pct
        if self.fast_weight_breakdown is not None:
            payload["fast_weight_breakdown"] = self.fast_weight_breakdown.as_dict()
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class LoopKpiMetrics:
    """Understanding loop KPI roll-up for the report."""

    regime_accuracy: float | None
    total_regime_evaluations: int
    drift_false_positive_rate: float | None
    drift_false_negative_rate: float | None
    drift_counts: Mapping[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "total_regime_evaluations": self.total_regime_evaluations,
            "drift_counts": dict(self.drift_counts),
        }
        if self.regime_accuracy is not None:
            payload["regime_accuracy"] = self.regime_accuracy
        if self.drift_false_positive_rate is not None:
            payload["drift_false_positive_rate"] = self.drift_false_positive_rate
        if self.drift_false_negative_rate is not None:
            payload["drift_false_negative_rate"] = self.drift_false_negative_rate
        return payload


@dataclass(frozen=True)
class StrategyPerformanceAggregates:
    """Portfolio-level aggregates across all strategies."""

    trades: int
    wins: int
    losses: int
    total_pnl: float
    total_notional: float
    roi: float
    win_rate: float
    average_return: float | None
    max_drawdown: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "total_pnl": self.total_pnl,
            "total_notional": self.total_notional,
            "roi": self.roi,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
        }
        if self.average_return is not None:
            payload["average_return"] = self.average_return
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class StrategyPerformanceReport:
    """Comprehensive KPI report for trading and understanding loops."""

    generated_at: datetime
    period_start: datetime
    strategies: Sequence[StrategyKpi]
    aggregates: StrategyPerformanceAggregates
    loop_metrics: LoopKpiMetrics
    roi_snapshot: RoiTelemetrySnapshot | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "period_start": self.period_start.astimezone(UTC).isoformat(),
            "strategies": [strategy.as_dict() for strategy in self.strategies],
            "aggregates": self.aggregates.as_dict(),
            "loop_metrics": self.loop_metrics.as_dict(),
        }
        if self.roi_snapshot is not None:
            payload["roi_snapshot"] = self.roi_snapshot.as_dict()
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def to_markdown(self) -> str:
        lines = [
            f"**Strategy KPIs** — generated at {self.generated_at.astimezone(UTC).isoformat()}",
            "",
            "| Strategy | Trades | Win rate | ROI | PnL | Max drawdown |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for strategy in self.strategies:
            max_dd = f"{strategy.max_drawdown:,.2f}"
            if strategy.max_drawdown_pct is not None:
                max_dd = f"{max_dd} ({strategy.max_drawdown_pct:.2%})"
            lines.append(
                "| {sid} | {trades} | {win_rate:.1%} | {roi:.2%} | {pnl:,.2f} | {max_dd} |".format(
                    sid=strategy.strategy_id,
                    trades=strategy.trades,
                    win_rate=strategy.win_rate,
                    roi=strategy.roi,
                    pnl=strategy.total_pnl,
                    max_dd=max_dd,
                )
            )

        lines.append("")
        agg = self.aggregates
        lines.append("**Portfolio totals**")
        lines.append(
            "- Trades: {trades} (wins: {wins}, losses: {losses})".format(
                trades=agg.trades, wins=agg.wins, losses=agg.losses
            )
        )
        lines.append(f"- ROI: {agg.roi:.2%} | Win rate: {agg.win_rate:.2%}")
        lines.append(f"- PnL: {agg.total_pnl:,.2f} | Max drawdown: {agg.max_drawdown:,.2f}")

        if self.loop_metrics.total_regime_evaluations:
            lines.append("")
            lines.append("**Understanding loop KPIs**")
            if self.loop_metrics.regime_accuracy is not None:
                lines.append(
                    f"- Regime accuracy: {self.loop_metrics.regime_accuracy:.2%}"
                )
            if self.loop_metrics.drift_false_positive_rate is not None:
                lines.append(
                    f"- Drift false-positive rate: {self.loop_metrics.drift_false_positive_rate:.2%}"
                )
            if self.loop_metrics.drift_false_negative_rate is not None:
                lines.append(
                    f"- Drift false-negative rate: {self.loop_metrics.drift_false_negative_rate:.2%}"
                )
        if self.roi_snapshot is not None:
            lines.append("")
            lines.append("**ROI posture**")
            lines.append(
                "- Status: {status} | ROI: {roi:.2%} | Annualised: {annual:.2%}".format(
                    status=self.roi_snapshot.status.value,
                    roi=self.roi_snapshot.roi,
                    annual=self.roi_snapshot.annualised_roi,
                )
            )
            lines.append(
                "- Net PnL: {net:,.2f} | Trades: {trades}".format(
                    net=self.roi_snapshot.net_pnl,
                    trades=self.roi_snapshot.executed_trades,
                )
            )
        return "\n".join(lines)


@dataclass
class _TradeRecord:
    timestamp: datetime
    pnl: float
    notional: float
    return_pct: float | None
    fast_weights_enabled: bool | None
    regime: str | None
    metadata: Mapping[str, Any]
    fast_weight_metrics: Mapping[str, Any] | None = None
    fast_weight_summary: Mapping[str, Mapping[str, Any]] | None = None
    baseline_return: float | None = None
    baseline_pnl: float | None = None


@dataclass
class _StrategyModeAccumulator:
    trades: int = 0
    pnl: float = 0.0
    notional: float = 0.0
    returns: list[float] = field(default_factory=list)

    def add(self, pnl: float, notional: float, return_pct: float | None) -> None:
        self.trades += 1
        self.pnl += pnl
        self.notional += abs(notional)
        if return_pct is not None:
            self.returns.append(return_pct)

    def summary(self) -> StrategyModeSummary | None:
        if self.trades == 0:
            return None
        roi = (self.pnl / self.notional) if self.notional else 0.0
        average_return = fmean(self.returns) if self.returns else None
        return StrategyModeSummary(
            trades=self.trades,
            pnl=self.pnl,
            notional=self.notional,
            roi=roi,
            average_return=average_return,
        )


@dataclass
class _StrategyAccumulator:
    strategy_id: str
    trades: list[_TradeRecord] = field(default_factory=list)
    wins: int = 0
    losses: int = 0
    pnl_total: float = 0.0
    notional_total: float = 0.0
    returns: list[float] = field(default_factory=list)
    fast_weight_modes: MutableMapping[bool, _StrategyModeAccumulator] = field(
        default_factory=lambda: {True: _StrategyModeAccumulator(), False: _StrategyModeAccumulator()}
    )
    metadata: Counter[str] = field(default_factory=Counter)
    fast_weight_toggle_counts: Counter[str] = field(default_factory=Counter)
    fast_weight_telemetry: _FastWeightTelemetryAccumulator = field(
        default_factory=_FastWeightTelemetryAccumulator
    )
    hebbian_adapters: MutableMapping[str, _HebbianAdapterAccumulator] = field(default_factory=dict)

    def add_trade(self, record: _TradeRecord) -> None:
        self.trades.append(record)
        self.pnl_total += record.pnl
        self.notional_total += abs(record.notional)
        if record.return_pct is not None:
            self.returns.append(record.return_pct)
        if record.pnl > 0:
            self.wins += 1
        elif record.pnl < 0:
            self.losses += 1
        if record.fast_weights_enabled is not None:
            self.fast_weight_modes.setdefault(
                record.fast_weights_enabled, _StrategyModeAccumulator()
            ).add(record.pnl, record.notional, record.return_pct)
            toggle_bucket = "enabled" if record.fast_weights_enabled else "disabled"
            self.fast_weight_toggle_counts[toggle_bucket] += 1
        else:
            self.fast_weight_toggle_counts["unknown"] += 1
        if record.regime:
            self.metadata[f"regime:{record.regime}"] += 1
        if record.fast_weight_metrics:
            self.fast_weight_telemetry.add_from_mapping(record.fast_weight_metrics)
        if record.fast_weight_summary:
            for adapter_id, summary in record.fast_weight_summary.items():
                if not isinstance(summary, Mapping):
                    continue
                adapter_key = str(adapter_id)
                adapter_accumulator = self.hebbian_adapters.setdefault(
                    adapter_key,
                    _HebbianAdapterAccumulator(adapter_id=adapter_key),
                )
                adapter_accumulator.add(summary)

    def kpi(self) -> StrategyKpi:
        trades_count = len(self.trades)
        win_rate = (self.wins / trades_count) if trades_count else 0.0
        roi = (self.pnl_total / self.notional_total) if self.notional_total else 0.0
        average_return = fmean(self.returns) if self.returns else None
        max_drawdown, max_drawdown_pct = self._max_drawdown()

        enabled_summary = self.fast_weight_modes.get(True, _StrategyModeAccumulator()).summary()
        disabled_summary = self.fast_weight_modes.get(False, _StrategyModeAccumulator()).summary()
        roi_uplift: float | None = None
        if enabled_summary and disabled_summary:
            roi_uplift = enabled_summary.roi - disabled_summary.roi

        metadata: dict[str, Any] = {}
        if self.metadata:
            metadata["regime_counts"] = dict(self.metadata)

        fast_weight_metadata = self._fast_weight_metadata()
        if fast_weight_metadata:
            metadata["fast_weight"] = fast_weight_metadata

        return StrategyKpi(
            strategy_id=self.strategy_id,
            trades=trades_count,
            wins=self.wins,
            losses=self.losses,
            total_pnl=self.pnl_total,
            total_notional=self.notional_total,
            roi=roi,
            win_rate=win_rate,
            average_return=average_return,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            fast_weight_breakdown=StrategyFastWeightBreakdown(
                enabled=enabled_summary,
                disabled=disabled_summary,
                roi_uplift=roi_uplift,
            )
            if enabled_summary or disabled_summary
            else None,
            metadata=metadata,
        )

    def _max_drawdown(self) -> tuple[float, float | None]:
        if not self.trades:
            return 0.0, None
        cumulative = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for record in sorted(self.trades, key=lambda trade: trade.timestamp):
            cumulative += record.pnl
            peak = max(peak, cumulative)
            drawdown = peak - cumulative
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        base = self.notional_total
        max_drawdown_pct = (max_drawdown / base) if base else None
        return max_drawdown, max_drawdown_pct

    def _fast_weight_metadata(self) -> Mapping[str, Any]:
        payload: dict[str, Any] = {}

        toggle_counts = {
            bucket: int(count)
            for bucket, count in self.fast_weight_toggle_counts.items()
            if count
        }
        if toggle_counts:
            payload["toggle_counts"] = toggle_counts

        metrics_summary = self.fast_weight_telemetry.summary()
        if metrics_summary:
            payload["metrics"] = dict(metrics_summary)

        hebbian_payload: dict[str, Any] = {}
        for adapter_id, accumulator in self.hebbian_adapters.items():
            summary = accumulator.summary()
            if summary:
                hebbian_payload[adapter_id] = dict(summary)
        if hebbian_payload:
            payload["hebbian_adapters"] = hebbian_payload

        return payload


def aggregate_fast_weight_metadata(
    events: Iterable[Mapping[str, Any]] | None,
) -> Mapping[str, Any]:
    """Summarise fast-weight telemetry embedded in experiment events."""

    if not events:
        return {}

    toggle_counts: Counter[str] = Counter()
    telemetry = _FastWeightTelemetryAccumulator()
    adapters: MutableMapping[str, _HebbianAdapterAccumulator] = {}

    for event in events:
        if not isinstance(event, Mapping):
            continue
        metadata = event.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        payload = metadata.get("fast_weight")
        if not isinstance(payload, Mapping):
            continue

        enabled = payload.get("enabled")
        if enabled is True:
            toggle_counts["enabled"] += 1
        elif enabled is False:
            toggle_counts["disabled"] += 1
        else:
            toggle_counts["unknown"] += 1

        metrics_payload = payload.get("metrics")
        if isinstance(metrics_payload, Mapping):
            telemetry.add_from_mapping(metrics_payload)

        summary_payload = payload.get("summary")
        if isinstance(summary_payload, Mapping):
            for adapter_id, adapter_summary in summary_payload.items():
                if not isinstance(adapter_summary, Mapping):
                    continue
                adapter_key = str(adapter_id)
                accumulator = adapters.setdefault(
                    adapter_key,
                    _HebbianAdapterAccumulator(adapter_id=adapter_key),
                )
                accumulator.add(adapter_summary)

    result: dict[str, Any] = {}
    if toggle_counts:
        result["toggle_counts"] = {
            bucket: int(count) for bucket, count in toggle_counts.items() if count
        }

    metrics_summary = telemetry.summary()
    if metrics_summary:
        result["metrics"] = dict(metrics_summary)

    if adapters:
        adapter_payload: dict[str, Any] = {}
        for adapter_id, accumulator in adapters.items():
            summary = accumulator.summary()
            if summary:
                adapter_payload[adapter_id] = dict(summary)
        if adapter_payload:
            result["hebbian_adapters"] = adapter_payload

    return result


class StrategyPerformanceTracker:
    """Collects per-strategy KPIs and exports daily reports."""

    def __init__(
        self,
        *,
        initial_capital: float,
        roi_cost_model: RoiCostModel | None = None,
        period_start: datetime | None = None,
        baseline_window: int = 20,
        baseline_min_trades: int = 5,
        baseline_tolerance_ratio: float = 0.25,
        baseline_min_gap: float = 0.0001,
        baseline_default_spread_bps: float = 1.0,
    ) -> None:
        self._initial_capital = max(0.0, float(initial_capital))
        self._cost_model = roi_cost_model or RoiCostModel.bootstrap_defaults(self._initial_capital)
        self._period_start = period_start or datetime.now(tz=UTC)
        self._strategies: MutableMapping[str, _StrategyAccumulator] = {}
        self._global_trades: list[_TradeRecord] = []
        self._regime_total = 0
        self._regime_correct = 0
        self._drift_counts: Counter[str] = Counter()
        self._fast_weight_toggle_counts: Counter[str] = Counter()
        self._fast_weight_telemetry = _FastWeightTelemetryAccumulator()
        self._hebbian_adapters: MutableMapping[str, _HebbianAdapterAccumulator] = {}
        self._baseline_window = max(1, int(baseline_window))
        self._baseline_min_trades = max(1, int(baseline_min_trades))
        self._baseline_tolerance_ratio = max(0.0, float(baseline_tolerance_ratio))
        self._baseline_min_gap = max(0.0, float(baseline_min_gap))
        self._baseline_default_spread_bps = max(0.0, float(baseline_default_spread_bps))
        self._baseline_default_return = (
            self._baseline_default_spread_bps / 10_000.0
            if self._baseline_default_spread_bps > 0.0
            else None
        )

    @property
    def period_start(self) -> datetime:
        return self._period_start

    def record_trade(
        self,
        strategy_id: str,
        *,
        pnl: float,
        notional: float,
        return_pct: float | None = None,
        timestamp: datetime | None = None,
        regime: str | None = None,
        fast_weights_enabled: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
        fast_weight_metrics: Mapping[str, Any] | None = None,
        fast_weight_summary: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        """Record a completed trade for KPI calculations."""

        if timestamp is None:
            timestamp = datetime.now(tz=UTC)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        pnl_value = float(pnl)
        notional_value = float(notional)
        notional_abs = abs(notional_value)
        metrics_payload = dict(fast_weight_metrics) if fast_weight_metrics else None
        summary_payload: Mapping[str, Mapping[str, Any]] | None = None
        if fast_weight_summary:
            summary_payload = {
                str(adapter_id): dict(summary)
                for adapter_id, summary in fast_weight_summary.items()
                if isinstance(summary, Mapping)
            }

        metadata_payload: Mapping[str, Any] = dict(metadata or {})

        if return_pct is None and notional_abs > 0.0:
            implied_return = pnl_value / notional_abs
            return_pct_value = implied_return
        else:
            return_pct_value = float(return_pct) if return_pct is not None else None

        baseline_return = self._infer_baseline_return(metadata_payload)
        baseline_pnl = (
            baseline_return * notional_abs if baseline_return is not None and notional_abs > 0.0 else None
        )
        if baseline_return is not None and isinstance(metadata_payload, dict):
            metadata_payload.setdefault("baseline_return_estimate", baseline_return)

        record = _TradeRecord(
            timestamp=timestamp.astimezone(UTC),
            pnl=pnl_value,
            notional=notional_value,
            return_pct=return_pct_value,
            fast_weights_enabled=fast_weights_enabled,
            regime=regime,
            metadata=metadata_payload,
            fast_weight_metrics=metrics_payload,
            fast_weight_summary=summary_payload,
            baseline_return=baseline_return,
            baseline_pnl=baseline_pnl,
        )
        accumulator = self._strategies.setdefault(
            strategy_id, _StrategyAccumulator(strategy_id=strategy_id)
        )
        accumulator.add_trade(record)
        self._global_trades.append(record)

        if fast_weights_enabled is True:
            self._fast_weight_toggle_counts["enabled"] += 1
        elif fast_weights_enabled is False:
            self._fast_weight_toggle_counts["disabled"] += 1
        else:
            self._fast_weight_toggle_counts["unknown"] += 1

        if metrics_payload:
            self._fast_weight_telemetry.add_from_mapping(metrics_payload)

        if summary_payload:
            for adapter_id, summary in summary_payload.items():
                adapter_accumulator = self._hebbian_adapters.setdefault(
                    adapter_id,
                    _HebbianAdapterAccumulator(adapter_id=adapter_id),
                )
                adapter_accumulator.add(summary)

    def _infer_baseline_return(self, metadata: Mapping[str, Any] | None) -> float | None:
        """Estimate a baseline fractional return using spread metadata."""

        default = self._baseline_default_return
        if not metadata:
            return default

        def _clip(value: float | None) -> float | None:
            if value is None or not math.isfinite(value):
                return None
            return max(0.0, min(value, 1.0))

        for key in (
            "baseline_expected_return",
            "baseline_return",
            "expected_return",
            "target_return",
        ):
            candidate = _clip(_coerce_ratio(metadata.get(key)))
            if candidate is not None:
                return candidate

        for key in ("spread_return_pct", "spread_pct", "mean_revert_return"):
            candidate = _clip(_coerce_ratio(metadata.get(key)))
            if candidate is not None:
                return candidate

        for key in ("spread_bps", "baseline_spread_bps", "expected_spread_bps"):
            bps_value = _coerce_float(metadata.get(key))
            if bps_value is not None:
                candidate = _clip(bps_value / 10_000.0)
                if candidate is not None:
                    return candidate

        for key in ("spread_pips", "baseline_spread_pips"):
            pips_value = _coerce_float(metadata.get(key))
            if pips_value is not None:
                candidate = _clip(pips_value / 10_4)
                if candidate is not None:
                    return candidate

        price_denominator = None
        for key in ("mid_price", "reference_price", "entry_price", "price"):
            value = _coerce_float(metadata.get(key))
            if value not in (None, 0.0):
                price_denominator = abs(value)
                break

        tick_size = None
        for key in ("tick_size", "tick_value", "price_increment"):
            value = _coerce_float(metadata.get(key))
            if value not in (None, 0.0):
                tick_size = abs(value)
                break

        for key in ("spread_ticks", "edge_ticks", "baseline_edge_ticks"):
            ticks = _coerce_float(metadata.get(key))
            if ticks is None:
                continue
            if tick_size is not None and price_denominator not in (None, 0.0):
                candidate = _clip(abs(ticks) * tick_size / price_denominator)
                if candidate is not None:
                    return candidate

        for key in ("spread", "bid_ask_spread", "price_spread"):
            spread_abs = _coerce_float(metadata.get(key))
            if spread_abs is None:
                continue
            if price_denominator not in (None, 0.0):
                candidate = _clip(abs(spread_abs) / price_denominator)
                if candidate is not None:
                    return candidate

        return default

    def _baseline_summary_for_records(
        self,
        records: Sequence[_TradeRecord],
        *,
        scope: str,
        identifier: str | None = None,
    ) -> Mapping[str, Any]:
        if not records:
            return {}

        window_size = min(self._baseline_window, len(records))
        recent = list(records[-window_size:])

        total_actual_pnl = 0.0
        total_baseline_pnl = 0.0
        total_notional = 0.0
        weighted_actual_return = 0.0
        weighted_baseline_return = 0.0
        samples = 0
        missing = 0
        start_ts: datetime | None = None
        end_ts: datetime | None = None

        for trade in recent:
            notional_abs = abs(trade.notional)
            if notional_abs <= 0.0:
                missing += 1
                continue

            actual_return = trade.return_pct
            if actual_return is None:
                actual_return = trade.pnl / notional_abs

            baseline_return = trade.baseline_return
            if baseline_return is None:
                baseline_return = self._baseline_default_return

            if baseline_return is None or actual_return is None:
                missing += 1
                continue

            baseline_pnl = trade.baseline_pnl
            if baseline_pnl is None:
                baseline_pnl = baseline_return * notional_abs

            total_actual_pnl += trade.pnl
            total_baseline_pnl += baseline_pnl
            total_notional += notional_abs
            weighted_actual_return += actual_return * notional_abs
            weighted_baseline_return += baseline_return * notional_abs
            samples += 1

            if start_ts is None or trade.timestamp < start_ts:
                start_ts = trade.timestamp
            if end_ts is None or trade.timestamp > end_ts:
                end_ts = trade.timestamp

        if samples == 0 or total_notional <= 0.0:
            return {}

        avg_actual_return = weighted_actual_return / total_notional
        avg_baseline_return = weighted_baseline_return / total_notional
        return_gap = avg_baseline_return - avg_actual_return
        pnl_shortfall = total_baseline_pnl - total_actual_pnl
        tolerance_value = self._baseline_tolerance_ratio * total_baseline_pnl

        window_minutes = 0.0
        if start_ts is not None and end_ts is not None and end_ts > start_ts:
            window_minutes = (end_ts - start_ts).total_seconds() / 60.0

        alert = False
        alert_reason = None
        if (
            samples >= self._baseline_min_trades
            and total_baseline_pnl > 0.0
            and return_gap >= self._baseline_min_gap
            and total_actual_pnl + tolerance_value < total_baseline_pnl
        ):
            alert = True
            alert_reason = "actual_below_baseline"

        summary: dict[str, Any] = {
            "scope": scope,
            "window_size": window_size,
            "window_trades": samples,
            "missing_samples": missing,
            "actual_pnl": total_actual_pnl,
            "baseline_pnl": total_baseline_pnl,
            "pnl_shortfall": pnl_shortfall,
            "actual_return": avg_actual_return,
            "baseline_return": avg_baseline_return,
            "return_gap": return_gap,
            "tolerance_ratio": self._baseline_tolerance_ratio,
            "min_gap": self._baseline_min_gap,
            "tolerance_value": tolerance_value,
            "window_minutes": window_minutes,
            "alert": alert,
        }

        if identifier is not None:
            summary["strategy_id" if scope == "strategy" else "identifier"] = identifier

        if alert_reason is not None:
            summary["alert_reason"] = alert_reason

        if end_ts is not None:
            summary["latest_trade_at"] = end_ts.astimezone(UTC).isoformat()
        if start_ts is not None:
            summary["earliest_trade_at"] = start_ts.astimezone(UTC).isoformat()

        return summary

    def _baseline_summary_for_strategy(self, strategy_id: str) -> Mapping[str, Any]:
        accumulator = self._strategies.get(strategy_id)
        if accumulator is None:
            return {}
        return self._baseline_summary_for_records(
            accumulator.trades,
            scope="strategy",
            identifier=strategy_id,
        )

    def _baseline_summary_for_portfolio(self) -> Mapping[str, Any]:
        return self._baseline_summary_for_records(
            self._global_trades,
            scope="portfolio",
            identifier="portfolio",
        )

    def record_regime_evaluation(self, predicted: str, actual: str) -> None:
        """Capture a regime classification outcome for accuracy KPIs."""

        self._regime_total += 1
        if predicted == actual:
            self._regime_correct += 1

    def record_drift_evaluation(self, *, triggered: bool, drift_present: bool) -> None:
        """Record a drift alert outcome for false positive/negative metrics."""

        self._drift_counts["total"] += 1
        if triggered and drift_present:
            self._drift_counts["true_positive"] += 1
        elif triggered and not drift_present:
            self._drift_counts["false_positive"] += 1
        elif not triggered and drift_present:
            self._drift_counts["false_negative"] += 1
        else:
            self._drift_counts["true_negative"] += 1

    def generate_report(self, *, as_of: datetime | None = None) -> StrategyPerformanceReport:
        """Compute KPIs and return a structured report."""

        generated_at = as_of or datetime.now(tz=UTC)
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=UTC)

        strategies_list = [acc.kpi() for acc in self._strategies.values()]
        baseline_alerts: list[Mapping[str, Any]] = []
        for idx, strategy in enumerate(strategies_list):
            summary = self._baseline_summary_for_strategy(strategy.strategy_id)
            if not summary:
                continue
            metadata = dict(strategy.metadata)
            metadata["baseline_comparator"] = summary
            strategies_list[idx] = replace(strategy, metadata=metadata)
            if summary.get("alert"):
                baseline_alerts.append(
                    {
                        "strategy_id": strategy.strategy_id,
                        "return_gap": summary.get("return_gap"),
                        "pnl_shortfall": summary.get("pnl_shortfall"),
                        "window_trades": summary.get("window_trades"),
                        "window_minutes": summary.get("window_minutes"),
                        "latest_trade_at": summary.get("latest_trade_at"),
                    }
                )

        strategies = tuple(strategies_list)
        aggregates = self._build_aggregates(strategies)
        loop_metrics = self._build_loop_metrics()
        roi_snapshot = self._build_roi_snapshot(generated_at)

        metadata = {
            "period_days": max((generated_at - self._period_start).total_seconds() / 86400.0, 0.0),
            "initial_capital": self._initial_capital,
        }

        portfolio_baseline = self._baseline_summary_for_portfolio()
        if portfolio_baseline:
            metadata["baseline_comparator"] = portfolio_baseline

        if baseline_alerts:
            metadata["baseline_alerts"] = baseline_alerts

        return StrategyPerformanceReport(
            generated_at=generated_at,
            period_start=self._period_start,
            strategies=strategies,
            aggregates=aggregates,
            loop_metrics=loop_metrics,
            roi_snapshot=roi_snapshot,
            metadata=metadata,
        )

    def _build_aggregates(self, strategies: Sequence[StrategyKpi]) -> StrategyPerformanceAggregates:
        trades = sum(strategy.trades for strategy in strategies)
        wins = sum(strategy.wins for strategy in strategies)
        losses = sum(strategy.losses for strategy in strategies)
        total_pnl = sum(strategy.total_pnl for strategy in strategies)
        total_notional = sum(strategy.total_notional for strategy in strategies)
        roi = (total_pnl / total_notional) if total_notional else 0.0
        win_rate = (wins / trades) if trades else 0.0

        returns: list[float] = []
        for strategy in strategies:
            if strategy.average_return is not None and strategy.trades:
                returns.extend([strategy.average_return] * strategy.trades)
        average_return = fmean(returns) if returns else None

        max_drawdown = self._global_max_drawdown()

        metadata: dict[str, Any] = {}
        fast_weight_entries: list[float] = []
        for strategy in strategies:
            breakdown = strategy.fast_weight_breakdown
            if breakdown and breakdown.roi_uplift is not None:
                fast_weight_entries.append(breakdown.roi_uplift)
        if fast_weight_entries:
            metadata["fast_weight_roi_uplift_mean"] = fmean(fast_weight_entries)

        fast_weight_metadata: dict[str, Any] = {}
        toggle_counts = {
            bucket: int(count)
            for bucket, count in self._fast_weight_toggle_counts.items()
            if count
        }
        if toggle_counts:
            fast_weight_metadata["toggle_counts"] = toggle_counts

        metrics_summary = self._fast_weight_telemetry.summary()
        if metrics_summary:
            fast_weight_metadata["metrics"] = dict(metrics_summary)

        hebbian_payload: dict[str, Any] = {}
        for adapter_id, accumulator in self._hebbian_adapters.items():
            summary = accumulator.summary()
            if summary:
                hebbian_payload[adapter_id] = dict(summary)
        if hebbian_payload:
            fast_weight_metadata["hebbian_adapters"] = hebbian_payload

        if fast_weight_metadata:
            metadata["fast_weight"] = fast_weight_metadata

        return StrategyPerformanceAggregates(
            trades=trades,
            wins=wins,
            losses=losses,
            total_pnl=total_pnl,
            total_notional=total_notional,
            roi=roi,
            win_rate=win_rate,
            average_return=average_return,
            max_drawdown=max_drawdown,
            metadata=metadata,
        )

    def _build_loop_metrics(self) -> LoopKpiMetrics:
        regime_accuracy: float | None = None
        if self._regime_total:
            regime_accuracy = self._regime_correct / self._regime_total

        false_positive_rate: float | None = None
        false_negative_rate: float | None = None
        if self._drift_counts:
            fp = self._drift_counts.get("false_positive", 0)
            tn = self._drift_counts.get("true_negative", 0)
            tp = self._drift_counts.get("true_positive", 0)
            fn = self._drift_counts.get("false_negative", 0)
            if (fp + tn) > 0:
                false_positive_rate = fp / (fp + tn)
            if (fn + tp) > 0:
                false_negative_rate = fn / (fn + tp)

        return LoopKpiMetrics(
            regime_accuracy=regime_accuracy,
            total_regime_evaluations=self._regime_total,
            drift_false_positive_rate=false_positive_rate,
            drift_false_negative_rate=false_negative_rate,
            drift_counts=dict(self._drift_counts),
        )

    def _build_roi_snapshot(self, generated_at: datetime) -> RoiTelemetrySnapshot | None:
        if not self._strategies:
            portfolio_state = {
                "equity": self._initial_capital,
                "total_pnl": 0.0,
            }
            return evaluate_roi_posture(
                portfolio_state,
                self._cost_model,
                executed_trades=0,
                total_notional=0.0,
                period_start=self._period_start,
                as_of=generated_at,
            )

        total_pnl = sum(record.pnl for record in self._global_trades)
        total_notional = sum(abs(record.notional) for record in self._global_trades)
        executed_trades = len(self._global_trades)

        portfolio_state = {
            "equity": self._initial_capital + total_pnl,
            "total_pnl": total_pnl,
        }

        return evaluate_roi_posture(
            portfolio_state,
            self._cost_model,
            executed_trades=executed_trades,
            total_notional=total_notional,
            period_start=self._period_start,
            as_of=generated_at,
        )

    def _global_max_drawdown(self) -> float:
        if not self._global_trades:
            return 0.0
        cumulative = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for record in sorted(self._global_trades, key=lambda trade: trade.timestamp):
            cumulative += record.pnl
            peak = max(peak, cumulative)
            drawdown = peak - cumulative
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        return max_drawdown


__all__ = [
    "StrategyModeSummary",
    "StrategyFastWeightBreakdown",
    "StrategyKpi",
    "LoopKpiMetrics",
    "StrategyPerformanceAggregates",
    "StrategyPerformanceReport",
    "aggregate_fast_weight_metadata",
    "StrategyPerformanceTracker",
]
