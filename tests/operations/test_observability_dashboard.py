from __future__ import annotations

from datetime import datetime, timedelta
from typing import Mapping

try:  # Python 3.10 compatibility
    from datetime import UTC
except ImportError:  # pragma: no cover - fallback for older runtimes
    from datetime import timezone

    UTC = timezone.utc

import pytest

from src.governance.policy_graduation import (
    DiaryMetrics,
    DiaryStageMetrics,
    PolicyGraduationAssessment,
)
from src.governance.policy_ledger import PolicyLedgerStage
from src.operations.data_backbone import (
    BackboneComponentSnapshot,
    BackboneStatus,
    DataBackboneReadinessSnapshot,
)
from src.operations.event_bus_health import (
    EventBusHealthSnapshot,
    EventBusHealthStatus,
)
from src.operations.evolution_kpis import (
    BudgetUsageKpi,
    EvolutionKpiSnapshot,
    EvolutionKpiStatus,
    PromotionKpi,
    RollbackLatencyKpi,
    TimeToCandidateKpi,
)
from src.operations.operator_leverage import (
    OperatorExperimentStats,
    OperatorLeverageSnapshot,
    OperatorLeverageStatus,
)
import src.operations.observability_dashboard as dashboard_module
from src.operations.observability_dashboard import (
    DashboardPanel,
    DashboardStatus,
    build_observability_dashboard,
)
from src.orchestration.alpha_trade_loop import (
    ComplianceEvent,
    ComplianceEventType,
    ComplianceSeverity,
)
from src.operations.operational_readiness import (
    OperationalReadinessComponent,
    OperationalReadinessSnapshot,
    OperationalReadinessStatus,
)
from src.operations.quality_telemetry import QualityStatus, QualityTelemetrySnapshot
from src.operations.roi import RoiStatus, RoiTelemetrySnapshot
from src.operations.slo import OperationalSLOSnapshot, ServiceSLO, SLOStatus
from src.risk.analytics.expected_shortfall import ExpectedShortfallResult
from src.risk.analytics.var import VarResult
from src.understanding import (
    UnderstandingDiagnosticsBuilder,
    UnderstandingGraphStatus,
    UnderstandingLoopSnapshot,
)
from src.thinking.adaptation.policy_reflection import PolicyReflectionArtifacts


pytestmark = pytest.mark.guardrail


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _roi_snapshot() -> RoiTelemetrySnapshot:
    moment = _now()
    return RoiTelemetrySnapshot(
        status=RoiStatus.tracking,
        generated_at=moment,
        initial_capital=100_000.0,
        current_equity=101_200.0,
        gross_pnl=1_800.0,
        net_pnl=1_500.0,
        infrastructure_cost=200.0,
        fees=100.0,
        days_active=10.0,
        executed_trades=48,
        total_notional=250_000.0,
        roi=0.015,
        annualised_roi=0.45,
        gross_roi=0.018,
        gross_annualised_roi=0.54,
        breakeven_daily_return=0.0003,
        target_annual_roi=0.25,
    )


def _event_bus_snapshot() -> EventBusHealthSnapshot:
    moment = _now()
    return EventBusHealthSnapshot(
        service="runtime",
        generated_at=moment,
        status=EventBusHealthStatus.warn,
        expected=True,
        running=True,
        loop_running=True,
        queue_size=120,
        queue_capacity=500,
        subscriber_count=6,
        topic_subscribers={"telemetry": 3},
        published_events=420,
        dropped_events=2,
        handler_errors=1,
        uptime_seconds=3600.0,
        last_event_at=moment - timedelta(seconds=30),
        last_error_at=None,
        issues=("Queue backlog observed",),
        metadata={"region": "primary"},
    )


def _slo_snapshot() -> OperationalSLOSnapshot:
    moment = _now()
    slos = (
        ServiceSLO(
            name="daily_bars",
            status=SLOStatus.met,
            message="Freshness within target",
            target={"freshness_seconds": 300},
            observed={"freshness_seconds": 120, "rows": 1200},
        ),
        ServiceSLO(
            name="intraday_trades",
            status=SLOStatus.at_risk,
            message="Lag approaching threshold",
            target={"freshness_seconds": 120},
            observed={"freshness_seconds": 150, "rows": 4800},
        ),
    )
    return OperationalSLOSnapshot(
        service="timescale_ingest",
        generated_at=moment,
        status=SLOStatus.at_risk,
        slos=slos,
        metadata={"region": "primary"},
    )


def _backbone_snapshot() -> DataBackboneReadinessSnapshot:
    moment = _now()
    components = (
        BackboneComponentSnapshot(
            name="Timescale ingest",
            status=BackboneStatus.warn,
            summary="Freshness outside SLO",
        ),
        BackboneComponentSnapshot(
            name="Redis cache",
            status=BackboneStatus.ok,
            summary="Nominal",
        ),
    )
    return DataBackboneReadinessSnapshot(
        status=BackboneStatus.warn,
        generated_at=moment,
        components=components,
        metadata={"environment": "paper"},
    )


def _operational_readiness_snapshot() -> OperationalReadinessSnapshot:
    moment = _now()
    components = (
        OperationalReadinessComponent(
            name="system_validation",
            status=OperationalReadinessStatus.ok,
            summary="All guardrails green",
        ),
        OperationalReadinessComponent(
            name="incident_response",
            status=OperationalReadinessStatus.warn,
            summary="Open incidents=1, missing runbooks=0",
        ),
    )
    return OperationalReadinessSnapshot(
        status=OperationalReadinessStatus.warn,
        generated_at=moment,
        components=components,
        metadata={"region": "primary"},
    )


def _quality_snapshot() -> QualityTelemetrySnapshot:
    moment = _now()
    return QualityTelemetrySnapshot(
        generated_at=moment,
        status=QualityStatus.warn,
        coverage_percent=78.0,
        coverage_target=80.0,
        staleness_hours=10.0,
        max_staleness_hours=24.0,
        notes=("Coverage 78.00% (target 80.00%)", "Coverage telemetry age 10.0h (max 24.0h)"),
        remediation_items=("Add ingest backfill tests",),
        metadata={"trend_counts": {"coverage_trend": 5, "coverage_domain_trend": 2, "remediation_trend": 1}},
    )


def _diary_metrics(
    policy_id: str,
    *,
    stage_metrics: Mapping[PolicyLedgerStage, DiaryStageMetrics],
    total_decisions: int,
    forced_total: int,
    severity_counts: Mapping[str, int],
    consecutive_normal: int,
    latest_stage: PolicyLedgerStage,
) -> DiaryMetrics:
    now = _now()
    release_routes = {stage.value: metrics.total for stage, metrics in stage_metrics.items()}
    return DiaryMetrics(
        policy_id=policy_id,
        total_decisions=total_decisions,
        stage_metrics=stage_metrics,
        release_routes=release_routes,
        severity_counts=severity_counts,
        forced_total=forced_total,
        consecutive_normal_latest_stage=consecutive_normal,
        first_entry_at=now - timedelta(hours=total_decisions),
        last_entry_at=now,
        latest_stage=latest_stage,
    )


def test_build_dashboard_composes_panels_and_status() -> None:
    roi_snapshot = _roi_snapshot()
    var_result = VarResult(value=12_000.0, confidence=0.99, sample_size=252)
    es_result = ExpectedShortfallResult(value=8_500.0, confidence=0.99, sample_size=252)

    dashboard = build_observability_dashboard(
        roi_snapshot=roi_snapshot,
        risk_results={
            "parametric_var": var_result,
            "expected_shortfall": es_result,
        },
        risk_limits={"parametric_var": 10_000.0, "expected_shortfall": 12_000.0},
        event_bus_snapshot=_event_bus_snapshot(),
        slo_snapshot=_slo_snapshot(),
        backbone_snapshot=_backbone_snapshot(),
        operational_readiness_snapshot=_operational_readiness_snapshot(),
        quality_snapshot=_quality_snapshot(),
        metadata={"environment": "paper"},
    )

    assert dashboard.status is DashboardStatus.fail
    assert {panel.name for panel in dashboard.panels} == {
        "PnL & ROI",
        "Risk & exposure",
        "Compliance & governance",
        "Evolution KPIs",
        "Operator leverage",
        "Latency & throughput",
        "System health",
        "Operational readiness",
        "Quality & coverage",
        "Understanding loop",
    }

    risk_panel = next(
        panel for panel in dashboard.panels if panel.name == "Risk & exposure"
    )
    assert risk_panel.status is DashboardStatus.fail
    assert "parametric_var" in risk_panel.metadata

    compliance_panel = next(
        panel for panel in dashboard.panels if panel.name == "Compliance & governance"
    )
    assert compliance_panel.status is DashboardStatus.warn
    assert compliance_panel.metadata["counts"]["risk_breach"] == 0
    assert "No compliance telemetry supplied" in compliance_panel.details[0]

    roi_panel = next(
        panel for panel in dashboard.panels if panel.name == "PnL & ROI"
    )
    assert roi_panel.status is DashboardStatus.ok
    assert roi_panel.metadata["roi"]["status"] == RoiStatus.tracking.value

    evolution_panel = next(
        panel for panel in dashboard.panels if panel.name == "Evolution KPIs"
    )
    assert evolution_panel.status is DashboardStatus.warn
    assert "Evolution KPI" in evolution_panel.headline

    latency_panel = next(
        panel for panel in dashboard.panels if panel.name == "Latency & throughput"
    )
    assert latency_panel.status is DashboardStatus.warn

    quality_panel = next(
        panel for panel in dashboard.panels if panel.name == "Quality & coverage"
    )
    assert quality_panel.status is DashboardStatus.warn
    assert "Coverage 78.00%" in quality_panel.headline
    assert "Add ingest backfill tests" in "\n".join(quality_panel.details)
    assert "event_bus" in latency_panel.metadata
    assert "slos" in latency_panel.metadata

    system_panel = next(
        panel for panel in dashboard.panels if panel.name == "System health"
    )
    assert system_panel.status is DashboardStatus.warn
    assert system_panel.metadata["backbone"]["status"] == BackboneStatus.warn.value

    readiness_panel = next(
        panel
        for panel in dashboard.panels
        if panel.name == "Operational readiness"
    )
    assert readiness_panel.status is DashboardStatus.warn
    assert readiness_panel.metadata["operational_readiness"]["status"] == (
        OperationalReadinessStatus.warn.value
    )
    assert any("incident_response" in detail for detail in readiness_panel.details)

    markdown = dashboard.to_markdown()
    assert "# Operational observability dashboard" in markdown
    assert "Risk & exposure" in markdown

    metadata_counts = dashboard.metadata["panel_status_counts"]
    assert metadata_counts == {
        DashboardStatus.ok.value: 1,
        DashboardStatus.warn.value: 8,
        DashboardStatus.fail.value: 1,
    }
    metadata_statuses = dashboard.metadata["panel_statuses"]
    assert metadata_statuses["Risk & exposure"] == DashboardStatus.fail.value
    assert metadata_statuses["Operational readiness"] == DashboardStatus.warn.value
    assert metadata_statuses["Compliance & governance"] == DashboardStatus.warn.value
    assert metadata_statuses["Operator leverage"] == DashboardStatus.warn.value
    assert metadata_statuses["Understanding loop"] == DashboardStatus.warn.value
    assert metadata_statuses["Evolution KPIs"] == DashboardStatus.warn.value

    remediation = dashboard.remediation_summary()
    assert remediation["overall_status"] == DashboardStatus.fail.value
    assert remediation["panel_counts"][DashboardStatus.fail.value] == 1
    assert remediation["panel_counts"][DashboardStatus.warn.value] == 8
    assert remediation["panel_counts"][DashboardStatus.ok.value] == 1
    assert remediation["failing_panels"] == ("Risk & exposure",)
    assert set(remediation["warning_panels"]) == {
        "Latency & throughput",
        "System health",
        "Operational readiness",
        "Quality & coverage",
        "Compliance & governance",
        "Understanding loop",
        "Evolution KPIs",
        "Operator leverage",
    }
    assert remediation["healthy_panels"] == ("PnL & ROI",)


def test_compliance_panel_records_breach_event() -> None:
    occurred_at = _now()
    breach_event = ComplianceEvent(
        event_type=ComplianceEventType.risk_breach,
        severity=ComplianceSeverity.critical,
        summary="Trade blocked due to risk breach",
        occurred_at=occurred_at,
        policy_id="alpha.paper",
        metadata={"reason": "limit_exceeded"},
    )

    class _LoopResultStub:
        def __init__(self, events: tuple[ComplianceEvent, ...]) -> None:
            self.compliance_events = events

    dashboard = build_observability_dashboard(
        loop_results=[_LoopResultStub((breach_event,))],
    )

    compliance_panel = next(
        panel for panel in dashboard.panels if panel.name == "Compliance & governance"
    )

    assert compliance_panel.status is DashboardStatus.fail
    assert compliance_panel.metadata["counts"]["risk_breach"] == 1
    event_payload = compliance_panel.metadata["events"][0]
    assert event_payload["summary"] == "Trade blocked due to risk breach"
    assert event_payload["policy_id"] == "alpha.paper"
    assert compliance_panel.metadata["compliance_slo"]["metadata"]["breaches"] == 1


def test_dashboard_handles_missing_inputs() -> None:
    dashboard = build_observability_dashboard()

    assert dashboard.status is DashboardStatus.warn
    assert len(dashboard.panels) == 4
    panels_by_name = {panel.name: panel for panel in dashboard.panels}

    compliance_panel = panels_by_name["Compliance & governance"]
    assert compliance_panel.status is DashboardStatus.warn
    assert "No compliance telemetry supplied" in compliance_panel.details[0]

    evolution_panel = panels_by_name["Evolution KPIs"]
    assert evolution_panel.status is DashboardStatus.warn
    assert "Evolution KPI" in evolution_panel.headline

    operator_panel = panels_by_name["Operator leverage"]
    assert operator_panel.status is DashboardStatus.warn
    assert "telemetry unavailable" in operator_panel.headline.lower()

    understanding_panel = panels_by_name["Understanding loop"]
    assert understanding_panel.status is DashboardStatus.warn
    assert "unavailable" in understanding_panel.headline.lower()
    assert "graph diagnostics" in understanding_panel.details[0].lower()

    markdown = dashboard.to_markdown()
    assert "| Panel" in markdown
    assert "Compliance & governance" in markdown
    assert "Understanding loop" in markdown

    payload = dashboard.as_dict()
    assert payload["status"] == DashboardStatus.warn.value
    assert {item["name"] for item in payload["panels"]} == {
        "Compliance & governance",
        "Evolution KPIs",
        "Operator leverage",
        "Understanding loop",
    }
    assert payload["metadata"]["panel_status_counts"] == {
        DashboardStatus.ok.value: 0,
        DashboardStatus.warn.value: 4,
        DashboardStatus.fail.value: 0,
    }
    assert payload["metadata"]["panel_statuses"] == {
        "Compliance & governance": DashboardStatus.warn.value,
        "Evolution KPIs": DashboardStatus.warn.value,
        "Operator leverage": DashboardStatus.warn.value,
        "Understanding loop": DashboardStatus.warn.value,
    }

    remediation = payload["remediation_summary"]
    assert remediation["overall_status"] == DashboardStatus.warn.value
    assert remediation["panel_counts"][DashboardStatus.warn.value] == 4
    assert remediation["panel_counts"][DashboardStatus.ok.value] == 0
    assert remediation["panel_counts"][DashboardStatus.fail.value] == 0
    assert set(remediation["warning_panels"]) == {
        "Compliance & governance",
        "Evolution KPIs",
        "Operator leverage",
        "Understanding loop",
    }
    assert remediation["healthy_panels"] == ()
    assert remediation["failing_panels"] == ()

    remediation_via_method = dashboard.remediation_summary()
    assert remediation_via_method == remediation


def test_dashboard_merges_additional_panels() -> None:
    custom_panel = DashboardPanel(
        name="Custom",
        status=DashboardStatus.warn,
        headline="Follow-up required",
    )

    dashboard = build_observability_dashboard(additional_panels=[custom_panel])

    assert dashboard.status is DashboardStatus.warn
    assert len(dashboard.panels) == 5
    assert dashboard.panels[0] is custom_panel
    panels_by_name = {panel.name: panel for panel in dashboard.panels[1:]}
    assert panels_by_name["Compliance & governance"].status is DashboardStatus.warn
    assert panels_by_name["Evolution KPIs"].status is DashboardStatus.warn
    assert panels_by_name["Operator leverage"].status is DashboardStatus.warn
    assert panels_by_name["Understanding loop"].status is DashboardStatus.warn
    assert dashboard.metadata["panel_status_counts"] == {
        DashboardStatus.ok.value: 0,
        DashboardStatus.warn.value: 5,
        DashboardStatus.fail.value: 0,
    }
    assert dashboard.metadata["panel_statuses"] == {
        "Custom": DashboardStatus.warn.value,
        "Compliance & governance": DashboardStatus.warn.value,
        "Evolution KPIs": DashboardStatus.warn.value,
        "Operator leverage": DashboardStatus.warn.value,
        "Understanding loop": DashboardStatus.warn.value,
    }


def test_evolution_panel_renders_snapshot() -> None:
    snapshot = EvolutionKpiSnapshot(
        generated_at=_now(),
        status=EvolutionKpiStatus.ok,
        time_to_candidate=TimeToCandidateKpi(
            count=8,
            average_hours=18.0,
            median_hours=12.0,
            p90_hours=24.0,
            max_hours=28.0,
            threshold_hours=24.0,
            sla_met=True,
            breaches=(),
        ),
        promotion=PromotionKpi(
            promotions=5,
            demotions=1,
            transitions=7,
            promotion_rate=5 / 7,
            recent_promotions=2,
            window_days=30.0,
            metadata={},
        ),
        budget=BudgetUsageKpi(
            samples=3,
            average_share=0.12,
            max_share=0.2,
            average_usage_ratio=0.6,
            max_usage_ratio=0.8,
            blocked_attempts=1,
            forced_decisions=0,
            metadata={},
        ),
        rollback=RollbackLatencyKpi(
            samples=2,
            average_hours=3.5,
            median_hours=3.0,
            p95_hours=3.8,
            max_hours=4.5,
            metadata={},
        ),
        metadata={"source": "unit-test"},
    )

    dashboard = build_observability_dashboard(evolution_kpis=snapshot)
    panel = next(panel for panel in dashboard.panels if panel.name == "Evolution KPIs")

    assert panel.status is DashboardStatus.ok
    joined_details = " ".join(panel.details)
    assert "Promotion posture" in joined_details
    assert "Exploration budget" in joined_details


def test_operator_leverage_panel_renders_snapshot() -> None:
    now = _now()
    snapshot = OperatorLeverageSnapshot(
        generated_at=now,
        status=OperatorLeverageStatus.warn,
        experiments_total=6,
        operator_count=2,
        weeks=2.0,
        experiments_per_week=2.5,
        experiments_per_week_total=5.0,
        quality_pass_rate=0.75,
        operators=(
            OperatorExperimentStats(
                operator="alice",
                experiments=4,
                experiments_per_week=2.0,
                quality_passes=3,
                quality_failures=1,
                quality_rate=0.75,
                last_experiment_at=now,
                recent_statuses=("executed", "executed"),
                metadata={"quality_events": 4},
            ),
            OperatorExperimentStats(
                operator="carol",
                experiments=2,
                experiments_per_week=1.0,
                quality_passes=1,
                quality_failures=1,
                quality_rate=0.5,
                last_experiment_at=now - timedelta(days=5),
                recent_statuses=("executed", "rejected"),
                metadata={"quality_events": 2},
            ),
        ),
        metadata={
            "lookback_days": 14.0,
            "low_velocity_warn": ("carol",),
            "quality_warn": ("alice",),
            "quality_missing": (),
            "top_failure_reasons": {"guardrail": 2},
        },
    )

    dashboard = build_observability_dashboard(operator_leverage_snapshot=snapshot)
    panel = next(panel for panel in dashboard.panels if panel.name == "Operator leverage")

    assert panel.status is DashboardStatus.warn
    assert "quality 75%" in panel.headline
    joined_details = " ".join(panel.details)
    assert "Top operators:" in joined_details
    assert "Velocity WARN" in joined_details
    assert "Quality WARN" in joined_details


def test_risk_panel_metadata_includes_limit_ratio() -> None:
    var_result = VarResult(value=9_000.0, confidence=0.99, sample_size=252)

    dashboard = build_observability_dashboard(
        risk_results={"parametric_var": var_result},
        risk_limits={"parametric_var": 10_000.0},
    )

    assert dashboard.status is DashboardStatus.warn

    risk_panel = next(
        panel for panel in dashboard.panels if panel.name == "Risk & exposure"
    )
    payload = risk_panel.metadata["parametric_var"]
    assert payload["limit"] == pytest.approx(10_000.0)
    assert payload["limit_ratio"] == pytest.approx(0.9)
    assert payload["limit_status"] == "warn"


def test_dashboard_risk_panel_announces_limit_statuses() -> None:
    dashboard = build_observability_dashboard(
        risk_results={
            "expected_shortfall": {"value": 1_200.0},
            "parametric_var": {"value": 90.0, "confidence": 0.99, "sample_size": 250},
        },
        risk_limits={"expected_shortfall": 0.0, "parametric_var": 100.0},
    )

    risk_panel = next(panel for panel in dashboard.panels if panel.name == "Risk & exposure")

    assert risk_panel.status is DashboardStatus.warn


def test_dashboard_includes_policy_graduation_panel_with_ready_and_blocked() -> None:
    ready_stage_metrics = {
        PolicyLedgerStage.EXPERIMENT: DiaryStageMetrics(
            stage=PolicyLedgerStage.EXPERIMENT,
            total=30,
            forced=3,
            severity_counts={"normal": 27, "warn": 3},
        ),
        PolicyLedgerStage.PAPER: DiaryStageMetrics(
            stage=PolicyLedgerStage.PAPER,
            total=20,
            forced=1,
            severity_counts={"normal": 19, "warn": 1},
        ),
    }
    ready_metrics = _diary_metrics(
        "alpha",
        stage_metrics=ready_stage_metrics,
        total_decisions=60,
        forced_total=6,
        severity_counts={"normal": 54, "warn": 6},
        consecutive_normal=18,
        latest_stage=PolicyLedgerStage.PAPER,
    )
    assessment_ready = PolicyGraduationAssessment(
        policy_id="alpha",
        current_stage=PolicyLedgerStage.EXPERIMENT,
        declared_stage=None,
        audit_stage=None,
        recommended_stage=PolicyLedgerStage.PAPER,
        metrics=ready_metrics,
        approvals=(),
        evidence_id="dd-alpha",
        audit_gaps=(),
        stage_blockers={
            PolicyLedgerStage.PAPER: (),
            PolicyLedgerStage.PILOT: ("insufficient_decisions",),
            PolicyLedgerStage.LIMITED_LIVE: ("insufficient_history",),
        },
    )

    blocked_stage_metrics = {
        PolicyLedgerStage.PAPER: DiaryStageMetrics(
            stage=PolicyLedgerStage.PAPER,
            total=50,
            forced=8,
            severity_counts={"normal": 42, "warn": 8},
        ),
        PolicyLedgerStage.PILOT: DiaryStageMetrics(
            stage=PolicyLedgerStage.PILOT,
            total=10,
            forced=2,
            severity_counts={"normal": 8, "warn": 2},
        ),
    }
    blocked_metrics = _diary_metrics(
        "beta",
        stage_metrics=blocked_stage_metrics,
        total_decisions=80,
        forced_total=12,
        severity_counts={"normal": 68, "warn": 12},
        consecutive_normal=9,
        latest_stage=PolicyLedgerStage.PILOT,
    )
    assessment_blocked = PolicyGraduationAssessment(
        policy_id="beta",
        current_stage=PolicyLedgerStage.PAPER,
        declared_stage=None,
        audit_stage=None,
        recommended_stage=PolicyLedgerStage.PILOT,
        metrics=blocked_metrics,
        approvals=("risk",),
        evidence_id="dd-beta",
        audit_gaps=("stale_evidence",),
        stage_blockers={
            PolicyLedgerStage.PAPER: (),
            PolicyLedgerStage.PILOT: ("approvals_required",),
            PolicyLedgerStage.LIMITED_LIVE: ("insufficient_history",),
        },
    )

    dashboard = build_observability_dashboard(
        policy_graduation_assessments=[assessment_ready, assessment_blocked]
    )

    panel = next(panel for panel in dashboard.panels if panel.name == "Policy graduation")
    assert panel.status is DashboardStatus.warn
    assert "alpha" in panel.details[0]
    assert "ready" in panel.details[0]
    assert "blocked" in panel.details[1]

    summary_meta = panel.metadata["policy_graduation"]["summary"]
    assert summary_meta["ready"] == ["alpha"]
    assert summary_meta["blocked"]["beta"] == ["approvals_required"]

    assert (
        dashboard.metadata["panel_statuses"]["Policy graduation"]
        == panel.status.value
    )


def test_policy_graduation_panel_marks_regression_as_failure() -> None:
    regression_stage_metrics = {
        PolicyLedgerStage.PILOT: DiaryStageMetrics(
            stage=PolicyLedgerStage.PILOT,
            total=45,
            forced=6,
            severity_counts={"normal": 36, "warn": 6, "alert": 3},
        ),
        PolicyLedgerStage.PAPER: DiaryStageMetrics(
            stage=PolicyLedgerStage.PAPER,
            total=30,
            forced=3,
            severity_counts={"normal": 27, "warn": 3},
        ),
    }
    regression_metrics = _diary_metrics(
        "gamma",
        stage_metrics=regression_stage_metrics,
        total_decisions=90,
        forced_total=12,
        severity_counts={"normal": 75, "warn": 12, "alert": 3},
        consecutive_normal=4,
        latest_stage=PolicyLedgerStage.PILOT,
    )
    assessment_regression = PolicyGraduationAssessment(
        policy_id="gamma",
        current_stage=PolicyLedgerStage.PILOT,
        declared_stage=None,
        audit_stage=None,
        recommended_stage=PolicyLedgerStage.PAPER,
        metrics=regression_metrics,
        approvals=("risk", "compliance"),
        evidence_id="dd-gamma",
        audit_gaps=(),
        stage_blockers={
            PolicyLedgerStage.PAPER: (),
            PolicyLedgerStage.LIMITED_LIVE: ("insufficient_history",),
        },
    )

    dashboard = build_observability_dashboard(
        policy_graduation_assessments=[assessment_regression]
    )

    panel = next(panel for panel in dashboard.panels if panel.name == "Policy graduation")
    assert panel.status is DashboardStatus.fail
    assert "regression" in panel.details[0].lower()

    summary_meta = panel.metadata["policy_graduation"]["summary"]
    assert summary_meta["regressing"] == [
        {
            "policy_id": "gamma",
            "current_stage": PolicyLedgerStage.PILOT.value,
            "recommended_stage": PolicyLedgerStage.PAPER.value,
        }
    ]


def test_understanding_panel_included_with_snapshot() -> None:
    builder = UnderstandingDiagnosticsBuilder(
        now=lambda: datetime(2024, 1, 1, tzinfo=UTC)
    )
    snapshot = builder.build().to_snapshot()

    dashboard = build_observability_dashboard(understanding_snapshot=snapshot)

    panel = next(panel for panel in dashboard.panels if panel.name == "Understanding loop")
    assert panel.status is DashboardStatus.ok

    payload = panel.metadata["understanding_loop"]
    assert payload["status"] == UnderstandingGraphStatus.ok.value
    assert payload["graph"]["metadata"]["decision_id"] == snapshot.decision.tactic_id


def test_understanding_panel_warns_when_snapshot_missing() -> None:
    dashboard = build_observability_dashboard()

    panel = next(panel for panel in dashboard.panels if panel.name == "Understanding loop")
    metadata = panel.metadata["understanding_loop"]

    assert panel.status is DashboardStatus.warn
    assert metadata["status"] == "missing"
    assert metadata["recommended_cli"].endswith("graph_diagnostics")


def test_understanding_panel_exports_throttle_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = UnderstandingDiagnosticsBuilder(now=lambda: datetime(2024, 1, 1, tzinfo=UTC))
    snapshot = builder.build().to_snapshot()

    calls: list[UnderstandingLoopSnapshot] = []
    monkeypatch.setattr(
        dashboard_module,
        "export_understanding_throttle_metrics",
        lambda value: calls.append(value),
    )

    build_observability_dashboard(understanding_snapshot=snapshot)

    assert calls == [snapshot]


def test_policy_reflection_panel_included_with_artifacts() -> None:
    digest = {
        "total_decisions": 4,
        "as_of": "2024-03-15T12:00:00+00:00",
        "tactics": [
            {
                "tactic_id": "breakout",
                "count": 3,
                "share": 0.75,
                "avg_score": 1.42,
                "last_seen": "2024-03-15T11:59:00+00:00",
            }
        ],
        "experiments": [
            {
                "experiment_id": "exp-boost",
                "count": 2,
                "share": 0.5,
                "last_seen": "2024-03-15T11:58:00+00:00",
                "regimes": ["bull"],
                "min_confidence": 0.6,
                "rationale": "Promote reversion in calm regimes",
            }
        ],
        "tags": [
            {
                "tag": "momentum",
                "count": 3,
                "share": 0.75,
            }
        ],
        "objectives": [
            {
                "objective": "alpha",
                "count": 3,
                "share": 0.75,
            }
        ],
        "recent_headlines": [
            "Selected breakout for bull (confidence=0.80)",
        ],
    }
    payload = {
        "metadata": {
            "generated_at": "2024-03-15T12:10:00+00:00",
            "total_decisions": 4,
            "as_of": "2024-03-15T12:00:00+00:00",
        },
        "digest": digest,
        "insights": (
            "Leading tactic breakout at 75.0% share (avg score 1.420)",
            "Top experiment exp-boost applied 2 times (50.0%)",
            "Dominant regime bull at 75.0%",
        ),
    }
    artifacts = PolicyReflectionArtifacts(
        digest=digest,
        markdown="# PolicyRouter reflection summary\n\nDecisions analysed: 4\n",
        payload=payload,
    )

    builder = UnderstandingDiagnosticsBuilder(
        now=lambda: datetime(2024, 3, 15, 12, 15, tzinfo=UTC)
    )
    snapshot = builder.build().to_snapshot()

    dashboard = build_observability_dashboard(
        understanding_snapshot=snapshot,
        policy_reflection=artifacts,
    )

    panel = next(panel for panel in dashboard.panels if panel.name == "Policy reflections")
    assert panel.status is DashboardStatus.ok
    assert "Top tactic breakout" in panel.details[0]
    assert any("Insight:" in detail for detail in panel.details)

    metadata = panel.metadata["policy_reflection"]
    assert metadata["metadata"]["total_decisions"] == 4
    assert metadata["markdown"].startswith("# PolicyRouter reflection summary")
    assert metadata["insights"][0].startswith("Leading tactic breakout")

    counts = dashboard.metadata["panel_status_counts"]
    assert counts[DashboardStatus.ok.value] >= 2
