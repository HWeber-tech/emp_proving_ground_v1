from datetime import datetime, timezone

from src.operations.compliance_readiness import (
    ComplianceReadinessStatus,
    evaluate_compliance_readiness,
)


def _trade_summary(status: str, *, failed: bool = False) -> dict[str, object]:
    checks: list[dict[str, object]] = [{"rule_id": "baseline", "passed": True, "severity": "info"}]
    if failed:
        checks.append({"rule_id": "limit", "passed": False, "severity": "critical"})
    return {
        "policy": {"policy_name": "inst-policy"},
        "last_snapshot": {
            "status": status,
            "checks": checks,
        },
        "daily_totals": {"EURUSD": {"notional": 1_000_000, "trades": 12}},
    }


def _kyc_summary(
    status: str,
    *,
    risk: str = "LOW",
    outstanding: int = 0,
    watchlist: int = 0,
    alerts: int = 0,
    open_cases: int = 0,
    escalations: int = 0,
    next_due: datetime | None = None,
) -> dict[str, object]:
    return {
        "last_snapshot": {
            "status": status,
            "risk_rating": risk,
            "outstanding_items": [f"item-{i}" for i in range(outstanding)],
            "watchlist_hits": [f"hit-{i}" for i in range(watchlist)],
            "alerts": [f"alert-{i}" for i in range(alerts)],
            "next_review_due": next_due.isoformat() if next_due else None,
        },
        "open_cases": open_cases,
        "escalations": escalations,
    }


def test_compliance_readiness_flags_trade_failures() -> None:
    snapshot = evaluate_compliance_readiness(trade_summary=_trade_summary("fail", failed=True))

    assert snapshot.status is ComplianceReadinessStatus.fail
    component = next(comp for comp in snapshot.components if comp.name == "trade_compliance")
    assert component.status is ComplianceReadinessStatus.fail
    assert component.metadata["critical_failures"] == 1


def test_compliance_readiness_ok_when_all_surfaces_clear() -> None:
    snapshot = evaluate_compliance_readiness(
        trade_summary=_trade_summary("pass"),
        kyc_summary=_kyc_summary("APPROVED"),
    )

    assert snapshot.status is ComplianceReadinessStatus.ok
    statuses = {component.name: component.status for component in snapshot.components}
    assert statuses["trade_compliance"] is ComplianceReadinessStatus.ok
    assert statuses["kyc_aml"] is ComplianceReadinessStatus.ok


def test_compliance_readiness_warns_on_kyc_outstanding_items() -> None:
    snapshot = evaluate_compliance_readiness(
        kyc_summary=_kyc_summary(
            "REVIEW_REQUIRED",
            outstanding=2,
            open_cases=1,
            next_due=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
    )

    assert snapshot.status is ComplianceReadinessStatus.warn
    kyc_component = next(comp for comp in snapshot.components if comp.name == "kyc_aml")
    assert kyc_component.status is ComplianceReadinessStatus.warn
    assert kyc_component.metadata["outstanding_items"] == 2
