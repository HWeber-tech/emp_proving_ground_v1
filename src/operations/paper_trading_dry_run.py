"""Evaluation helpers for sustained paper trading dry runs.

These utilities analyse :class:`PaperTradingSimulationReport` payloads produced
by ``run_paper_trading_simulation`` and emit a verdict describing whether the
session satisfied throttle, latency, and incident response expectations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, MutableMapping, Sequence

from src.operations.incident_response import IncidentResponseStatus
from src.runtime.paper_simulation import PaperTradingSimulationReport

__all__ = [
    "DryRunSeverity",
    "PaperDryRunIssue",
    "PaperDryRunBudgets",
    "PaperDryRunResult",
    "evaluate_paper_trading_dry_run",
]


_STATUS_ORDER: Mapping[IncidentResponseStatus, int] = {
    IncidentResponseStatus.ok: 0,
    IncidentResponseStatus.warn: 1,
    IncidentResponseStatus.fail: 2,
}


class DryRunSeverity(StrEnum):
    """Severity level for dry run findings."""

    pass_ = "pass"
    warn = "warn"
    fail = "fail"


@dataclass(frozen=True)
class PaperDryRunIssue:
    """Finding surfaced while evaluating a paper trading dry run."""

    severity: DryRunSeverity
    message: str
    context: Mapping[str, object] | None = None

    def as_dict(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "severity": self.severity.value,
            "message": self.message,
        }
        if self.context is not None:
            payload["context"] = dict(self.context)
        return payload


@dataclass(frozen=True)
class PaperDryRunBudgets:
    """Acceptance thresholds applied to paper trading dry runs."""

    max_avg_latency_s: float = 1.0
    max_failure_ratio: float = 0.05
    max_processing_ms: float = 250.0
    max_lag_ms: float = 250.0
    require_trade_throttle: bool = True
    require_incident_snapshot: bool = True
    max_incident_status: IncidentResponseStatus = IncidentResponseStatus.ok
    min_orders: int = 1


@dataclass(frozen=True)
class PaperDryRunResult:
    """Verdict returned after evaluating a paper trading dry run."""

    status: DryRunSeverity
    issues: tuple[PaperDryRunIssue, ...] = field(default_factory=tuple)
    metrics: Mapping[str, object] = field(default_factory=dict)

    def passed(self) -> bool:
        return self.status is DryRunSeverity.pass_

    def as_dict(self) -> Mapping[str, object]:
        return {
            "status": self.status.value,
            "issues": [issue.as_dict() for issue in self.issues],
            "metrics": dict(self.metrics),
        }


def evaluate_paper_trading_dry_run(
    report: PaperTradingSimulationReport,
    budgets: PaperDryRunBudgets | None = None,
) -> PaperDryRunResult:
    """Validate throttle, latency, and incident response posture for a dry run."""

    budgets = budgets or PaperDryRunBudgets()
    issues: list[PaperDryRunIssue] = []
    metrics: MutableMapping[str, object] = {
        "order_count": len(report.orders),
    }
    if report.paper_metrics is not None:
        metrics["paper_metrics"] = dict(report.paper_metrics)
    if report.execution_stats is not None:
        metrics["execution_stats"] = dict(report.execution_stats)
    if report.performance_health is not None:
        metrics["performance_health"] = dict(report.performance_health)
    if report.trade_throttle is not None:
        metrics["trade_throttle"] = dict(report.trade_throttle)
    if report.trade_throttle_scopes is not None:
        metrics["trade_throttle_scopes"] = [
            dict(scope) for scope in report.trade_throttle_scopes
        ]
    if report.incident_response is not None:
        metrics["incident_response"] = dict(report.incident_response)

    def _record(
        severity: DryRunSeverity,
        message: str,
        **context: object,
    ) -> None:
        payload = context or None
        issues.append(PaperDryRunIssue(severity=severity, message=message, context=payload))

    orders_observed = len(report.orders)
    if orders_observed < max(0, budgets.min_orders):
        _record(
            DryRunSeverity.fail,
            "Paper trading session did not reach the minimum order target",
            observed=orders_observed,
            expected=budgets.min_orders,
        )

    paper_metrics = report.paper_metrics or {}
    avg_latency = _coerce_float(paper_metrics.get("avg_latency_s"))
    if avg_latency is None:
        _record(
            DryRunSeverity.warn,
            "Paper broker metrics missing average latency",
        )
    elif avg_latency > budgets.max_avg_latency_s:
        _record(
            DryRunSeverity.fail,
            "Paper broker latency exceeded budget",
            observed=avg_latency,
            budget=budgets.max_avg_latency_s,
        )

    failure_ratio = _coerce_float(paper_metrics.get("failure_ratio"))
    if failure_ratio is None:
        _record(
            DryRunSeverity.warn,
            "Paper broker metrics missing failure ratio",
        )
    elif failure_ratio > budgets.max_failure_ratio:
        _record(
            DryRunSeverity.fail,
            "Paper broker failure ratio exceeded budget",
            observed=failure_ratio,
            budget=budgets.max_failure_ratio,
        )

    performance_health = report.performance_health or {}
    throughput = performance_health.get("throughput")
    if isinstance(throughput, Mapping):
        max_processing = _coerce_float(throughput.get("max_processing_ms"))
        if max_processing is not None and max_processing > budgets.max_processing_ms:
            _record(
                DryRunSeverity.fail,
                "Processing latency breached throughput budget",
                observed=max_processing,
                budget=budgets.max_processing_ms,
            )
        elif max_processing is None:
            _record(
                DryRunSeverity.warn,
                "Throughput metrics missing max processing measurement",
            )

        max_lag = _coerce_float(throughput.get("max_lag_ms"))
        if max_lag is not None and max_lag > budgets.max_lag_ms:
            _record(
                DryRunSeverity.fail,
                "Decision lag exceeded backlog threshold",
                observed=max_lag,
                budget=budgets.max_lag_ms,
            )
        elif max_lag is None:
            _record(
                DryRunSeverity.warn,
                "Throughput metrics missing max lag measurement",
            )
        elif not bool(throughput.get("healthy", True)):
            _record(
                DryRunSeverity.warn,
                "Throughput metrics flagged unhealthy posture",
            )
    else:
        _record(
            DryRunSeverity.warn,
            "Performance health snapshot missing throughput metrics",
        )

    if budgets.require_trade_throttle:
        if not report.trade_throttle:
            _record(
                DryRunSeverity.fail,
                "Trade throttle snapshot missing from dry run report",
            )
        else:
            state = str(report.trade_throttle.get("state", "")).strip()
            if not state:
                _record(
                    DryRunSeverity.warn,
                    "Trade throttle snapshot missing state information",
                )
        if report.trade_throttle_scopes is None:
            _record(
                DryRunSeverity.warn,
                "Trade throttle scope telemetry not captured",
            )

    if budgets.require_incident_snapshot:
        incident_payload = report.incident_response or {}
        snapshot = incident_payload.get("snapshot")
        if not isinstance(snapshot, Mapping):
            _record(
                DryRunSeverity.fail,
                "Incident response snapshot missing from dry run report",
            )
        else:
            status_raw = snapshot.get("status")
            try:
                status = (
                    status_raw
                    if isinstance(status_raw, IncidentResponseStatus)
                    else IncidentResponseStatus(str(status_raw))
                )
            except ValueError:
                _record(
                    DryRunSeverity.fail,
                    "Incident response snapshot reported unknown status",
                    observed=status_raw,
                )
            else:
                if _STATUS_ORDER[status] > _STATUS_ORDER[budgets.max_incident_status]:
                    _record(
                        DryRunSeverity.fail,
                        "Incident response status exceeded allowed severity",
                        observed=status.value,
                        allowed=budgets.max_incident_status.value,
                    )

    if any(issue.severity is DryRunSeverity.fail for issue in issues):
        status = DryRunSeverity.fail
    elif any(issue.severity is DryRunSeverity.warn for issue in issues):
        status = DryRunSeverity.warn
    else:
        status = DryRunSeverity.pass_

    return PaperDryRunResult(status=status, issues=tuple(issues), metrics=metrics)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None
