from __future__ import annotations

from pathlib import Path

import pytest

from tools.telemetry.pytest_guardrails import (
    PytestCase,
    analyse_report,
    evaluate_requirements,
    main,
    render_text_report,
)


@pytest.fixture
def junit_report(tmp_path: Path) -> Path:
    payload = """
    <testsuite name="pytest" tests="4" failures="1" skipped="1">
        <testcase classname="tests.data_foundation.test_timescale_backbone_orchestrator" name="test_ok" />
        <testcase classname="tests.trading.test_risk_policy" name="test_violation">
            <failure message="boom" />
        </testcase>
        <testcase classname="tests.trading.test_risk_policy" name="test_warning">
            <skipped message="slow" />
        </testcase>
        <testcase classname="tests.trading.test_risk_policy_telemetry" name="test_pass" />
    </testsuite>
    """.strip()
    report = tmp_path / "pytest-report.xml"
    report.write_text(payload, encoding="utf-8")
    return report


def test_analyse_report_parses_cases(junit_report: Path) -> None:
    cases = analyse_report(junit_report)
    assert cases == (
        PytestCase(
            classname="tests.data_foundation.test_timescale_backbone_orchestrator",
            name="test_ok",
            outcome="passed",
        ),
        PytestCase(
            classname="tests.trading.test_risk_policy",
            name="test_violation",
            outcome="failed",
        ),
        PytestCase(
            classname="tests.trading.test_risk_policy",
            name="test_warning",
            outcome="skipped",
        ),
        PytestCase(
            classname="tests.trading.test_risk_policy_telemetry",
            name="test_pass",
            outcome="passed",
        ),
    )


def test_evaluate_requirements_handles_missing_and_passing(junit_report: Path) -> None:
    cases = analyse_report(junit_report)
    results = evaluate_requirements(
        cases,
        [
            "tests.data_foundation.test_timescale_backbone_orchestrator",
            "tests.trading.test_risk_policy",
            "tests.trading.test_not_present",
            "tests.trading.test_risk_policy_telemetry::test_pass",
        ],
    )

    status_by_requirement = {result.requirement: result for result in results}

    assert status_by_requirement[
        "tests.data_foundation.test_timescale_backbone_orchestrator"
    ].satisfied
    assert not status_by_requirement["tests.trading.test_risk_policy"].satisfied
    assert status_by_requirement["tests.trading.test_risk_policy"].reason.startswith(
        "matching tests present but not passing"
    )
    assert not status_by_requirement["tests.trading.test_not_present"].satisfied
    assert status_by_requirement["tests.trading.test_not_present"].reason == "no matching tests found"
    assert status_by_requirement[
        "tests.trading.test_risk_policy_telemetry::test_pass"
    ].satisfied


def test_render_text_report_includes_status(junit_report: Path) -> None:
    cases = analyse_report(junit_report)
    results = evaluate_requirements(cases, ["tests.data_foundation.test_timescale_backbone_orchestrator"])
    text = render_text_report(results)
    assert "Pytest guardrail results" in text
    assert "PASS tests.data_foundation.test_timescale_backbone_orchestrator" in text


def test_cli_returns_failure_when_requirements_unsatisfied(junit_report: Path) -> None:
    code = main(
        [
            "--report",
            str(junit_report),
            "--require",
            "tests.trading.test_not_present",
        ]
    )
    assert code == 1


def test_cli_supports_json_output(junit_report: Path) -> None:
    code = main(
        [
            "--report",
            str(junit_report),
            "--require",
            "tests.data_foundation.test_timescale_backbone_orchestrator",
            "--format",
            "json",
        ]
    )
    assert code == 0
