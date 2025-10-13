from src.operations.incident_response import IncidentResponseStatus
from src.operations.paper_trading_dry_run import (
    DryRunSeverity,
    PaperDryRunBudgets,
    evaluate_paper_trading_dry_run,
)
from src.runtime.paper_simulation import PaperTradingSimulationReport


def _build_base_report(**overrides):
    payload = {
        "orders": [{"order_id": "1"}, {"order_id": "2"}],
        "paper_metrics": {
            "avg_latency_s": 0.2,
            "failure_ratio": 0.0,
        },
        "performance_health": {
            "throughput": {
                "max_processing_ms": 120.0,
                "max_lag_ms": 40.0,
                "healthy": True,
            }
        },
        "trade_throttle": {
            "state": "open",
            "metadata": {"max_trades": 10},
        },
        "trade_throttle_scopes": [
            {"scope": {"strategy_id": "bootstrap"}, "state": "open"}
        ],
        "incident_response": {
            "snapshot": {"status": IncidentResponseStatus.ok.value}
        },
    }
    payload.update(overrides)
    return PaperTradingSimulationReport(**payload)


def test_evaluate_paper_trading_dry_run_passes() -> None:
    report = _build_base_report()
    result = evaluate_paper_trading_dry_run(report)
    assert result.status is DryRunSeverity.pass_
    assert not result.issues
    assert result.passed() is True


def test_evaluate_paper_trading_dry_run_flags_latency() -> None:
    report = _build_base_report(
        paper_metrics={"avg_latency_s": 2.1, "failure_ratio": 0.0}
    )
    result = evaluate_paper_trading_dry_run(report)
    assert result.status is DryRunSeverity.fail
    assert any("latency" in issue.message.lower() for issue in result.issues)


def test_evaluate_paper_trading_dry_run_requires_incident_snapshot() -> None:
    report = _build_base_report(incident_response=None)
    result = evaluate_paper_trading_dry_run(report)
    assert result.status is DryRunSeverity.fail
    assert any("incident" in issue.message.lower() for issue in result.issues)


def test_evaluate_paper_trading_dry_run_allows_higher_incident_budget() -> None:
    report = _build_base_report(
        incident_response={
            "snapshot": {"status": IncidentResponseStatus.warn.value}
        }
    )
    budgets = PaperDryRunBudgets(max_incident_status=IncidentResponseStatus.warn)
    result = evaluate_paper_trading_dry_run(report, budgets=budgets)
    assert result.status is DryRunSeverity.pass_


def test_evaluate_paper_trading_dry_run_flags_incident_severity() -> None:
    report = _build_base_report(
        incident_response={"snapshot": {"status": IncidentResponseStatus.warn.value}}
    )
    result = evaluate_paper_trading_dry_run(report)
    assert result.status is DryRunSeverity.fail
    assert any("incident" in issue.message.lower() for issue in result.issues)
