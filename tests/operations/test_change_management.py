from datetime import UTC, datetime, timedelta

from src.operations.change_management import (
    ChangeApproval,
    ChangeAssessmentStatus,
    ChangeImpact,
    ChangeManagementAssessment,
    ChangeRequest,
    ChangeStatus,
    ChangeWindow,
    evaluate_change_request,
    generate_change_management_markdown,
)


def _make_window(start: datetime, *, hours: float) -> ChangeWindow:
    return ChangeWindow(start=start, end=start + timedelta(hours=hours))


def test_change_management_approves_low_risk_when_policy_met() -> None:
    now = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
    window = _make_window(now + timedelta(hours=8), hours=2)
    approvals = (
        ChangeApproval(
            approver="Olivia Operator",
            role="operations",
            approved_at=now + timedelta(hours=1),
        ),
    )
    request = ChangeRequest(
        change_id="CHG-101",
        title="Patch analytics cluster",
        description="Routine security patches during the overnight window.",
        impact=ChangeImpact.low,
        requested_by="Quentin",
        created_at=now,
        window=window,
        approvals=approvals,
        status=ChangeStatus.approved,
        tags=("maintenance", "analytics"),
    )

    assessment = evaluate_change_request(request, reference_time=now)

    assert assessment.status is ChangeAssessmentStatus.approved
    assert assessment.issues == ()
    assert assessment.warnings == ()
    assert assessment.approvals_missing == ()


def test_change_management_blocks_when_required_roles_missing() -> None:
    now = datetime(2024, 6, 1, 9, 0, tzinfo=UTC)
    window = _make_window(now + timedelta(days=2), hours=3)
    approvals = (
        ChangeApproval(
            approver="Riley Reviewer",
            role="operations",
            approved_at=now + timedelta(hours=2),
        ),
    )
    request = ChangeRequest(
        change_id="CHG-202",
        title="Upgrade market data routers",
        description="Network firmware upgrade across the redundant routers.",
        impact=ChangeImpact.high,
        requested_by="Sasha",
        created_at=now,
        window=window,
        approvals=approvals,
        status=ChangeStatus.pending_approval,
    )

    assessment = evaluate_change_request(request, reference_time=now)

    assert assessment.status is ChangeAssessmentStatus.blocked
    assert assessment.issues and "Missing required approvals" in assessment.issues[0]
    assert set(assessment.approvals_missing) == {"risk", "compliance"}


def test_change_management_flags_past_window_start() -> None:
    now = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
    window = ChangeWindow(
        start=now - timedelta(hours=1),
        end=now + timedelta(hours=1),
    )
    approvals = (
        ChangeApproval(
            approver="Casey Compliance",
            role="compliance",
            approved_at=now - timedelta(days=1),
        ),
        ChangeApproval(
            approver="Riley Risk",
            role="risk",
            approved_at=now - timedelta(days=1),
        ),
        ChangeApproval(
            approver="Olivia Operator",
            role="operations",
            approved_at=now - timedelta(days=1),
        ),
    )
    request = ChangeRequest(
        change_id="CHG-303",
        title="Deploy execution engine hotfix",
        description="Emergency hotfix rollout with follow-up validation.",
        impact=ChangeImpact.high,
        requested_by="Taylor",
        created_at=now - timedelta(days=3),
        window=window,
        approvals=approvals,
        status=ChangeStatus.scheduled,
    )

    assessment = evaluate_change_request(request, reference_time=now)

    assert assessment.status is ChangeAssessmentStatus.needs_attention
    assert assessment.issues == ()
    assert any("start is in the past" in warning for warning in assessment.warnings)


def test_change_management_detects_lead_time_violation() -> None:
    now = datetime(2024, 6, 1, 8, 0, tzinfo=UTC)
    window = _make_window(now + timedelta(hours=1), hours=2)
    approvals = (
        ChangeApproval(
            approver="Olivia Operator",
            role="operations",
            approved_at=now + timedelta(minutes=10),
        ),
        ChangeApproval(
            approver="Morgan Manager",
            role="team_owner",
            approved_at=now + timedelta(minutes=20),
        ),
    )
    request = ChangeRequest(
        change_id="CHG-404",
        title="Refresh staging environment",
        description="Short maintenance to recycle staging workloads.",
        impact=ChangeImpact.medium,
        requested_by="Jamie",
        created_at=now,
        window=window,
        approvals=approvals,
        status=ChangeStatus.scheduled,
    )

    assessment = evaluate_change_request(request, reference_time=now)

    assert assessment.status is ChangeAssessmentStatus.blocked
    assert any("Lead time" in issue for issue in assessment.issues)


def test_change_management_markdown_includes_key_sections() -> None:
    now = datetime(2024, 6, 1, 10, 0, tzinfo=UTC)
    window = _make_window(now + timedelta(hours=10), hours=2)
    approvals = (
        ChangeApproval(
            approver="Nina", role="operations", approved_at=now + timedelta(hours=1)
        ),
    )
    request = ChangeRequest(
        change_id="CHG-505",
        title="Rotate API credentials",
        description="Coordinated secret rotation across control plane services.",
        impact=ChangeImpact.low,
        requested_by="Morgan",
        created_at=now,
        window=window,
        approvals=approvals,
        status=ChangeStatus.approved,
    )

    assessment = ChangeManagementAssessment(
        status=ChangeAssessmentStatus.needs_attention,
        warnings=("Change window duration (2.0h) exceeds policy maximum of 1.0h",),
    )

    markdown = generate_change_management_markdown(request, assessment)

    assert "# Change Request CHG-505" in markdown
    assert "Rotate API credentials" in markdown
    assert "## Warnings" in markdown
    assert "operations: Nina" in markdown
    assert "secret rotation" in markdown

